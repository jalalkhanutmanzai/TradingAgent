import ccxt
import pandas as pd
import pytest

from data.fetcher import OHLCVFetcher, ohlcv_to_frame
from tests.conftest import make_ohlcv, raw_rows


class FakeExchange:
    """Minimal CCXT stand-in serving rows from an in-memory frame."""

    def __init__(self, df: pd.DataFrame, fail_times: int = 0):
        self.rows = raw_rows(df)
        self.fail_times = fail_times
        self.calls = 0

    def parse_timeframe(self, timeframe: str) -> int:
        return {"1m": 60, "1h": 3600}[timeframe]

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ccxt.NetworkError("boom")
        rows = [r for r in self.rows if since is None or r[0] >= since]
        if since is None:
            return rows[-limit:] if limit else rows
        return rows[:limit] if limit else rows

    def fetch_ticker(self, symbol):
        return {"last": self.rows[-1][4]}


def test_ohlcv_to_frame_sorts_and_dedups():
    rows = [[2000, 1, 2, 0.5, 1.5, 10], [1000, 1, 2, 0.5, 1.2, 10], [2000, 1, 2, 0.5, 1.5, 10]]
    df = ohlcv_to_frame(rows)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.index.is_monotonic_increasing
    assert str(df.index.tz) == "UTC"


def test_ohlcv_to_frame_empty():
    df = ohlcv_to_frame([])
    assert df.empty


def test_fetch_latest():
    src = make_ohlcv(50)
    f = OHLCVFetcher(FakeExchange(src))
    df = f.fetch_ohlcv("BTC/USDT", "1h", limit=10)
    assert len(df) == 10
    assert df["close"].iloc[-1] == pytest.approx(src["close"].iloc[-1])


def test_fetch_history_paginates():
    src = make_ohlcv(250)
    ex = FakeExchange(src)
    f = OHLCVFetcher(ex)
    since = int(src.index[0].timestamp() * 1000)
    df = f.fetch_history("BTC/USDT", "1h", since=since, page_limit=100)
    assert len(df) == 250
    assert ex.calls == 3


def test_fetch_history_until():
    src = make_ohlcv(100)
    f = OHLCVFetcher(FakeExchange(src))
    since = int(src.index[0].timestamp() * 1000)
    until = int(src.index[49].timestamp() * 1000)
    df = f.fetch_history("BTC/USDT", "1h", since=since, until=until, page_limit=30)
    assert len(df) == 50
    assert df.index[-1] == src.index[49]


def test_retries_transient_errors():
    ex = FakeExchange(make_ohlcv(20), fail_times=2)
    f = OHLCVFetcher(ex, max_retries=3, retry_delay=0)
    assert len(f.fetch_ohlcv("BTC/USDT", limit=5)) == 5
    assert ex.calls == 3


def test_gives_up_after_max_retries():
    ex = FakeExchange(make_ohlcv(20), fail_times=5)
    f = OHLCVFetcher(ex, max_retries=2, retry_delay=0)
    with pytest.raises(ccxt.NetworkError):
        f.fetch_ohlcv("BTC/USDT")


def test_fetch_last_price():
    src = make_ohlcv(20)
    assert OHLCVFetcher(FakeExchange(src)).fetch_last_price("BTC/USDT") == pytest.approx(src["close"].iloc[-1])
