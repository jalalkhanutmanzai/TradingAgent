"""Shared fixtures: deterministic synthetic OHLCV data and fake external clients."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import RiskConfig  # noqa: E402


def make_ohlcv(n: int = 200, drift: float = 0.0, start: float = 100.0,
               vol: float = 0.01, seed: int = 42) -> pd.DataFrame:
    """Geometric random walk with a deterministic seed; `drift` per bar (e.g. 0.003)."""
    rng = np.random.default_rng(seed)
    rets = drift + vol * rng.standard_normal(n)
    close = start * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start], close[:-1]])
    spread = np.abs(rng.standard_normal(n)) * vol * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(10, 100, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def raw_rows(df: pd.DataFrame) -> list[list[float]]:
    """Convert a frame back to CCXT-style [ms, o, h, l, c, v] rows."""
    ts = (df.index.as_unit("ms").asi8).tolist()
    return [[t, *vals] for t, vals in zip(ts, df[["open", "high", "low", "close", "volume"]].to_numpy().tolist())]


@pytest.fixture
def ohlcv_flat() -> pd.DataFrame:
    return make_ohlcv(drift=0.0)


@pytest.fixture
def ohlcv_uptrend() -> pd.DataFrame:
    return make_ohlcv(drift=0.004, vol=0.004)


@pytest.fixture
def ohlcv_downtrend() -> pd.DataFrame:
    # seed=1: default seed (42) has momentum decelerate right at the last bar,
    # leaving macd_hist positive despite the firm EMA downtrend.
    return make_ohlcv(drift=-0.004, vol=0.004, seed=1)


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(risk_per_trade=0.01, max_drawdown=0.10, max_position_pct=0.5,
                      min_reward_risk=1.5, min_confidence=0.55, max_open_positions=2)


class FakeMessages:
    """Stands in for client.beta.messages; records calls and returns canned results."""

    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def fake_anthropic_client(result=None, error: Exception | None = None):
    messages = FakeMessages(result=result, error=error)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def fake_parsed_response(parsed_output, stop_reason: str = "end_turn"):
    return SimpleNamespace(parsed_output=parsed_output, stop_reason=stop_reason, content=[])
