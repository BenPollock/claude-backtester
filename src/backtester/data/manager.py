"""DataManager: orchestrates cache-first data loading with gap-filling."""

import logging
from datetime import date

import pandas as pd

from backtester.data.cache import ParquetCache
from backtester.data.calendar import TradingCalendar
from backtester.data.sources.base import DataSource
from backtester.data.sources.yahoo import YahooDataSource

logger = logging.getLogger(__name__)

# Max consecutive NaN trading days before we stop forward-filling
MAX_FFILL_DAYS = 5


class DataManager:
    """Cache-first data loading with forward-fill for small gaps."""

    def __init__(self, cache_dir: str = "~/.backtester/cache", source: DataSource | None = None):
        self._cache = ParquetCache(cache_dir)
        self._source = source or YahooDataSource()
        self._calendar = TradingCalendar()

    def load(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Load OHLCV data for symbol, using cache when possible.

        Returns DataFrame indexed by trading days with columns:
        Open, High, Low, Close, Volume. Small gaps (<=5 days) are
        forward-filled; larger gaps remain NaN.
        """
        cached_range = self._cache.date_range(symbol)

        if cached_range is not None:
            cache_start, cache_end = cached_range
            if cache_start <= start and cache_end >= end:
                # Cache fully covers the range
                df = self._cache.load(symbol)
                return self._prepare(df, symbol, start, end)

        # Fetch from source (full range to keep cache complete)
        logger.info(f"Fetching {symbol} from source")
        df = self._source.fetch(symbol, start, end)
        df = self._cache.merge_and_save(symbol, df)
        return self._prepare(df, symbol, start, end)

    def _prepare(self, df: pd.DataFrame, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Trim to date range, reindex to trading days, and forward-fill gaps."""
        trading_days = self._calendar.trading_days(start, end)

        # Reindex to trading calendar
        df = df.reindex(trading_days)

        # Forward-fill gaps up to MAX_FFILL_DAYS
        df = df.ffill(limit=MAX_FFILL_DAYS)

        # Log remaining NaN gaps
        nan_count = df["Close"].isna().sum()
        if nan_count > 0:
            logger.warning(f"{symbol}: {nan_count} trading days with missing data after forward-fill")

        return df

    def load_many(self, symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
        """Load data for multiple symbols."""
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.load(symbol, start, end)
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
        return result
