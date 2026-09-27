import numpy as np
import pandas as pd
import pytest

from indicators.technical import add_all, atr, ema, macd, rsi, true_range


def test_ema_constant_series():
    s = pd.Series([5.0] * 30)
    out = ema(s, 10)
    assert out.iloc[:9].isna().all()
    assert np.allclose(out.dropna(), 5.0)


def test_ema_matches_recursive_definition():
    s = pd.Series(np.arange(1, 21, dtype=float))
    out = ema(s, 5)
    alpha = 2 / 6
    expected = s.iloc[0]
    for x in s.iloc[1:]:
        expected = alpha * x + (1 - alpha) * expected
    assert out.iloc[-1] == pytest.approx(expected)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 50, dtype=float))
    down = pd.Series(np.arange(50, 1, -1, dtype=float))
    flat = pd.Series([10.0] * 50)
    assert rsi(up).iloc[-1] == pytest.approx(100.0)
    assert rsi(down).iloc[-1] == pytest.approx(0.0)
    assert rsi(flat).iloc[-1] == pytest.approx(50.0)


def test_rsi_in_range(ohlcv_flat):
    r = rsi(ohlcv_flat["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert rsi(ohlcv_flat["close"]).iloc[:14].isna().all()


def test_macd_columns_and_hist(ohlcv_flat):
    m = macd(ohlcv_flat["close"])
    assert list(m.columns) == ["macd", "macd_signal", "macd_hist"]
    valid = m.dropna()
    assert np.allclose(valid["macd_hist"], valid["macd"] - valid["macd_signal"])


def test_macd_positive_in_uptrend(ohlcv_uptrend):
    assert macd(ohlcv_uptrend["close"])["macd"].iloc[-1] > 0


def test_macd_invalid_periods():
    with pytest.raises(ValueError):
        macd(pd.Series([1.0, 2.0]), fast=26, slow=12)


def test_true_range_uses_gaps():
    df = pd.DataFrame({"high": [10.0, 12.0], "low": [9.0, 11.5], "close": [9.5, 12.0]})
    tr = true_range(df)
    assert tr.iloc[0] == pytest.approx(1.0)
    assert tr.iloc[1] == pytest.approx(2.5)  # |high - prev_close| = 12 - 9.5


def test_atr_constant_range():
    n = 40
    df = pd.DataFrame({"high": [11.0] * n, "low": [9.0] * n, "close": [10.0] * n})
    assert atr(df, 14).iloc[-1] == pytest.approx(2.0)


def test_add_all(ohlcv_flat):
    out = add_all(ohlcv_flat)
    for col in ["ema_20", "ema_50", "rsi", "macd", "macd_signal", "macd_hist", "atr"]:
        assert col in out.columns
        assert not np.isnan(out[col].iloc[-1])
    assert "ema_20" not in ohlcv_flat.columns  # input not mutated


def test_add_all_requires_ohlcv():
    with pytest.raises(ValueError):
        add_all(pd.DataFrame({"close": [1.0, 2.0]}))


@pytest.mark.parametrize("fn", [ema, rsi])
def test_invalid_period(fn):
    with pytest.raises(ValueError):
        fn(pd.Series([1.0, 2.0]), 0)
