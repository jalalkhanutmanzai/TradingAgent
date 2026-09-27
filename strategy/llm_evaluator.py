"""LLM-guided signal evaluator backed by Claude structured outputs.

The model only *proposes* a trade. Every proposal is re-validated into a
TradeSignal (price geometry checks) and must still pass the RiskEngine.
Any API failure, refusal or malformed output fails closed to HOLD.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Literal

import anthropic
import pandas as pd
from pydantic import BaseModel, ValidationError

from config.settings import LLMConfig
from strategy.signal import Action, TradeSignal

logger = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You are a quantitative trading analyst. You receive a JSON snapshot of \
technical indicators for one market and must return a single trade decision.

Decide BUY, SELL or HOLD from trend (EMA fast vs slow), momentum (MACD histogram), \
overextension (RSI) and volatility (ATR). Prefer HOLD when signals conflict or conviction is low; \
a missed trade costs nothing, a bad trade costs capital.

For BUY: stop_loss < entry < take_profit. For SELL: take_profit < entry < stop_loss. \
Place the stop beyond recent noise, typically 1.5-3x ATR from entry, and target at least \
1.5x the stop distance. Use the latest close as entry unless you have a clear reason not to. \
For HOLD, set entry, stop_loss and take_profit to null.

confidence is your probability (0-1) that the trade reaches take_profit before stop_loss. \
Position sizing is handled downstream by a separate risk engine; do not reason about size. \
Keep the rationale to two or three sentences citing the indicators that drove the decision."""


class LLMSignalOutput(BaseModel):
    """Schema Claude must fill. Kept constraint-free; TradeSignal enforces the rules."""

    action: Literal["BUY", "SELL", "HOLD"]
    entry: float | None
    stop_loss: float | None
    take_profit: float | None
    confidence: float
    rationale: str


def build_market_snapshot(df: pd.DataFrame, symbol: str, lookback: int = 20) -> dict:
    """Summarise the latest indicator state plus recent closes for the prompt."""
    if df.empty:
        raise ValueError("cannot build snapshot from empty frame")
    last = df.iloc[-1]

    def _num(value) -> float | None:
        v = float(value)
        return None if math.isnan(v) else round(v, 8)

    indicator_cols = [c for c in df.columns if c not in ("open", "high", "low", "volume")]
    return {
        "symbol": symbol,
        "as_of": str(df.index[-1]),
        "latest": {c: _num(last[c]) for c in indicator_cols},
        "recent_closes": [round(float(x), 8) for x in df["close"].tail(lookback)],
        "recent_high": _num(df["high"].tail(lookback).max()),
        "recent_low": _num(df["low"].tail(lookback).min()),
    }


class LLMSignalEvaluator:
    def __init__(self, config: LLMConfig, client: anthropic.Anthropic | None = None):
        self.config = config
        self.client = client or anthropic.Anthropic()

    def _request_kwargs(self, snapshot: dict) -> dict:
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "thinking": {"type": "adaptive"},
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": json.dumps(snapshot, sort_keys=True)}],
            "output_format": LLMSignalOutput,
        }
        if self.config.use_fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def evaluate(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        try:
            snapshot = build_market_snapshot(df, symbol)
        except ValueError as exc:
            return TradeSignal.hold(symbol, f"no data: {exc}", "llm")

        try:
            response = self.client.beta.messages.parse(**self._request_kwargs(snapshot))
        except anthropic.RateLimitError as exc:
            logger.warning("LLM rate limited: %s", exc)
            return TradeSignal.hold(symbol, "LLM rate limited", "llm")
        except anthropic.APIStatusError as exc:
            logger.error("LLM API error %s: %s", exc.status_code, exc)
            return TradeSignal.hold(symbol, f"LLM API error {exc.status_code}", "llm")
        except anthropic.APIConnectionError as exc:
            logger.error("LLM connection error: %s", exc)
            return TradeSignal.hold(symbol, "LLM unreachable", "llm")

        if response.stop_reason == "refusal":
            return TradeSignal.hold(symbol, "LLM declined the request", "llm")
        if response.stop_reason == "max_tokens":
            return TradeSignal.hold(symbol, "LLM output truncated", "llm")

        out = response.parsed_output
        if out is None:
            return TradeSignal.hold(symbol, "LLM returned no structured output", "llm")

        try:
            return TradeSignal(
                symbol=symbol,
                action=Action(out.action),
                entry=out.entry,
                stop_loss=out.stop_loss,
                take_profit=out.take_profit,
                confidence=min(max(out.confidence, 0.0), 1.0),
                rationale=out.rationale,
                source="llm",
            )
        except ValidationError as exc:
            logger.warning("Rejected invalid LLM signal: %s", exc)
            return TradeSignal.hold(symbol, "LLM proposed invalid price levels", "llm")
