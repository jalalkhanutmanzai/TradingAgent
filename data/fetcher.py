"""OHLCV market data ingestion via CCXT."""
from __future__ import annotations

import logging
import time

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def ohlcv_to_frame(rows: list[list[float]]) -> pd.DataFrame:
    """Convert raw CCXT OHLCV rows into a UTC-indexed, de-duplicated DataFrame."""
    df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    if df.empty:
        return df.set_index(pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    return df.astype(float)


class OHLCVFetcher:
    """Fetches latest and historical candles, retrying on transient network errors."""

    def __init__(self, exchange: ccxt.Exchange, max_retries: int = 3, retry_delay: float = 1.0):
        self.exchange = exchange
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _call(self, fn, *args, **kwargs):
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.RateLimitExceeded) as exc:
                if attempt == self.max_retries:
                    raise
                logger.warning("Transient exchange error (%s), retry %d/%d", exc, attempt, self.max_retries)
                time.sleep(self.retry_delay * attempt)

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200,
                    since: int | None = None) -> pd.DataFrame:
        """Fetch the most recent `limit` candles (or from `since`, ms epoch)."""
        rows = self._call(self.exchange.fetch_ohlcv, symbol, timeframe, since=since, limit=limit)
        return ohlcv_to_frame(rows or [])

    def fetch_history(self, symbol: str, timeframe: str, since: int,
                      until: int | None = None, page_limit: int = 1000) -> pd.DataFrame:
        """Paginate forward from `since` until `until` (ms epoch) or no more data."""
        tf_ms = self.exchange.parse_timeframe(timeframe) * 1000
        all_rows: list[list[float]] = []
        cursor = since
        while True:
            rows = self._call(self.exchange.fetch_ohlcv, symbol, timeframe, since=cursor, limit=page_limit)
            if not rows:
                break
            all_rows.extend(rows)
            last_ts = rows[-1][0]
            if (until is not None and last_ts >= until) or len(rows) < page_limit:
                break
            cursor = last_ts + tf_ms
        df = ohlcv_to_frame(all_rows)
        if until is not None and not df.empty:
            df = df[df.index <= pd.Timestamp(until, unit="ms", tz="UTC")]
        return df

    def fetch_last_price(self, symbol: str) -> float:
        ticker = self._call(self.exchange.fetch_ticker, symbol)
        return float(ticker["last"])
