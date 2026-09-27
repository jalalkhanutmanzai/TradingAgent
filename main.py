"""Entry point: fetch data -> compute indicators -> evaluate signal -> risk-check -> execute.

Runs a single decision cycle end-to-end, executing in simulated paper-trading
mode by default. Wire `run_once` into a scheduler/loop for continuous trading;
it is kept single-pass here so each stage's outcome is easy to trace and test.
"""
from __future__ import annotations

import logging
import os
import sys

from config.broker import create_exchange
from config.settings import Settings
from data.fetcher import OHLCVFetcher
from execution.executor import BaseExecutor, ExecutionError, PaperExecutor
from indicators.technical import add_all
from risk.engine import RiskEngine
from strategy.llm_evaluator import LLMSignalEvaluator
from strategy.rule_based import RuleBasedEvaluator

logger = logging.getLogger(__name__)

DEFAULT_PAPER_EQUITY = float(os.getenv("PAPER_STARTING_EQUITY", "10000"))


def build_evaluator(settings: Settings):
    """Prefer the LLM evaluator when enabled and credentials are available;
    fail back to the deterministic rule-based evaluator otherwise."""
    if settings.llm.enabled and os.getenv("ANTHROPIC_API_KEY"):
        return LLMSignalEvaluator(settings.llm)
    logger.info("Using rule-based evaluator (LLM disabled or ANTHROPIC_API_KEY not set)")
    return RuleBasedEvaluator()


def run_once(settings: Settings | None = None, executor: BaseExecutor | None = None) -> None:
    settings = settings or Settings.from_env()
    exchange = create_exchange(settings.exchange)
    fetcher = OHLCVFetcher(exchange)
    evaluator = build_evaluator(settings)
    risk_engine = RiskEngine(settings.risk, DEFAULT_PAPER_EQUITY)
    executor = executor or PaperExecutor(risk_engine)

    df = fetcher.fetch_ohlcv(settings.symbol, settings.timeframe)
    if df.empty:
        logger.warning("No market data returned for %s", settings.symbol)
        return

    df = add_all(df)
    signal = evaluator.evaluate(df, settings.symbol)
    logger.info(
        "Signal: %s %s (confidence=%.2f) - %s",
        signal.action.value, signal.symbol, signal.confidence, signal.rationale,
    )

    decision = risk_engine.evaluate(signal)
    if not decision.approved:
        logger.info("Trade rejected: %s", "; ".join(decision.reasons))
        return

    try:
        order = executor.open_position(signal, decision.quantity)
    except ExecutionError as exc:
        logger.error("Execution failed: %s", exc)
        return

    logger.info(
        "Filled %s %s qty=%.6f @ %.2f (risk=%.2f, notional=%.2f)",
        order.side.value, order.symbol, order.quantity, order.price,
        decision.risk_amount, decision.notional,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run_once()
    except Exception:
        logger.exception("Fatal error during trading cycle")
        sys.exit(1)
