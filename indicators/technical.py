"""Technical indicators implemented with pandas (Wilder smoothing for RSI/ATR)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    if period < 1:
        raise ValueError("period must be >= 1")
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    if period < 1:
        raise ValueError("period must be >= 1")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # No losses -> RSI 100; flat market (no gains or losses) -> neutral 50.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out.where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    if fast >= slow:
        raise ValueError("fast period must be shorter than slow period")
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": macd_line - signal_line}
    )


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1, skipna=True)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if period < 1:
        raise ValueError("period must be >= 1")
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_all(df: pd.DataFrame, ema_fast: int = 20, ema_slow: int = 50,
            rsi_period: int = 14, atr_period: int = 14) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with all indicator columns appended."""
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {sorted(missing)}")
    out = df.copy()
    out[f"ema_{ema_fast}"] = ema(out["close"], ema_fast)
    out[f"ema_{ema_slow}"] = ema(out["close"], ema_slow)
    out["rsi"] = rsi(out["close"], rsi_period)
    out = out.join(macd(out["close"]))
    out["atr"] = atr(out, atr_period)
    return out.replace([np.inf, -np.inf], np.nan)
