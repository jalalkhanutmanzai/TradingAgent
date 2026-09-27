"""Deterministic trend-following evaluator; used standalone or as an LLM-free baseline."""
from __future__ import annotations

import math

import pandas as pd

from strategy.signal import Action, TradeSignal


class RuleBasedEvaluator:
    """EMA trend + MACD momentum, filtered by RSI extremes. SL/TP are ATR multiples."""

    def __init__(self, ema_fast: str = "ema_20", ema_slow: str = "ema_50",
                 sl_atr_mult: float = 2.0, tp_atr_mult: float = 3.0,
                 rsi_overbought: float = 70.0, rsi_oversold: float = 30.0):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def evaluate(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        cols = ["close", self.ema_fast, self.ema_slow, "rsi", "macd_hist", "atr"]
        if df.empty or any(c not in df.columns for c in cols):
            return TradeSignal.hold(symbol, "insufficient indicator data", "rule_based")
        last = df[cols].iloc[-1]
        if any(math.isnan(v) for v in last.to_numpy(dtype=float)) or last["atr"] <= 0:
            return TradeSignal.hold(symbol, "indicators not warmed up", "rule_based")

        close, atr_v = float(last["close"]), float(last["atr"])
        up = last[self.ema_fast] > last[self.ema_slow] and last["macd_hist"] > 0
        down = last[self.ema_fast] < last[self.ema_slow] and last["macd_hist"] < 0

        if up and last["rsi"] < self.rsi_overbought:
            return TradeSignal(
                symbol=symbol, action=Action.BUY, entry=close,
                stop_loss=close - self.sl_atr_mult * atr_v,
                take_profit=close + self.tp_atr_mult * atr_v,
                confidence=0.6, rationale="EMA uptrend with positive MACD momentum",
                source="rule_based",
            )
        if down and last["rsi"] > self.rsi_oversold and close - self.tp_atr_mult * atr_v > 0:
            return TradeSignal(
                symbol=symbol, action=Action.SELL, entry=close,
                stop_loss=close + self.sl_atr_mult * atr_v,
                take_profit=close - self.tp_atr_mult * atr_v,
                confidence=0.6, rationale="EMA downtrend with negative MACD momentum",
                source="rule_based",
            )
        return TradeSignal.hold(symbol, "no aligned trend/momentum setup", "rule_based")
