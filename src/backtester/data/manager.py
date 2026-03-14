"""DataManager: orchestrates cache-first data loading with gap-filling."""

import logging
from datetime import date

import pandas as pd

from backtester.data.cache import ParquetCache
from backtester.data.calendar import TradingCalendar
from backtester.data.sources.base import DataSource
from backtester.data.sources.yahoo import YahooDataSource

logger = logging.getLogger(__name__)

# Supported resampling timeframes beyond daily
_RESAMPLE_RULES = {
    "weekly": "W",
    "monthly": "ME",
}


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample daily OHLCV data to a lower frequency (weekly or monthly).

    Applies proper OHLCV aggregation rules:
        Open  = first non-NaN value in the period
        High  = max of the period
        Low   = min of the period
        Close = last non-NaN value in the period
        Volume = sum of the period

    The resulting index uses the last trading day of each period (not the
    calendar period-end) so it aligns cleanly with daily trading dates.

    Partial periods at the start or end of the data are included.

    Args:
        df: Daily OHLCV DataFrame with a DatetimeIndex.
        timeframe: One of 'weekly' or 'monthly'.

    Returns:
        Resampled DataFrame indexed by the period's last trading day.

    Raises:
        ValueError: If timeframe is not 'weekly' or 'monthly'.
    """
    if timeframe not in _RESAMPLE_RULES:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. Choose from: {sorted(_RESAMPLE_RULES)}"
        )

    df = df.copy()
    rule = _RESAMPLE_RULES[timeframe]

    agg_rules = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }

    resampled = df.resample(rule).agg(agg_rules).dropna(subset=["Close"])

    # Replace the period-end calendar date with the actual last trading day
    # in each period so the index aligns with daily trading dates.
    last_trading_days = df.resample(rule).apply(lambda x: x.index[-1] if len(x) > 0 else None)
    # last_trading_days is a Series; use the non-null entries as the new index
    if "Close" in last_trading_days.columns:
        # resample().apply on a DataFrame returns a DataFrame; use any column
        new_index = last_trading_days["Close"].dropna()
    else:
        new_index = last_trading_days.dropna()

    # Build mapping from period-end date to last trading day
    new_idx = []
    for period_end in resampled.index:
        # Find the last trading day <= period_end in the original data
        mask = df.index <= period_end
        if mask.any():
            new_idx.append(df.index[mask][-1])
        else:
            new_idx.append(period_end)

    resampled.index = pd.DatetimeIndex(new_idx)
    resampled.index.name = df.index.name

    return resampled

# Max consecutive NaN trading days before we stop forward-filling
MAX_FFILL_DAYS = 5


class DataManager:
    """Cache-first data loading with forward-fill for small gaps."""

    def __init__(self, cache_dir: str = "~/.backtester/cache", source: DataSource | None = None):
        self._cache = ParquetCache(cache_dir)
        self._source = source or YahooDataSource()
        self._calendar = TradingCalendar()

    # Minimum fraction of expected trading days that cached data must cover
    # after reindexing. Below this threshold the cache is considered stale
    # or polluted and the data is re-fetched from the source.
    _MIN_CACHE_COVERAGE = 0.90

    def load(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Load OHLCV data for symbol, using cache when possible.

        Returns DataFrame indexed by trading days with columns:
        Open, High, Low, Close, Volume. Small gaps (<=5 days) are
        forward-filled; larger gaps remain NaN.
        """
        cached_range = self._cache.date_range(symbol)

        if cached_range is not None:
            cache_start, cache_end = cached_range
            trading_days = self._calendar.trading_days(start, end)
            first_needed = trading_days[0].date() if len(trading_days) > 0 else start
            last_needed = trading_days[-1].date() if len(trading_days) > 0 else end
            if cache_start <= first_needed and cache_end >= last_needed:
                logger.info(f"Fetching {symbol} from cache")
                df = self._cache.load(symbol)
                prepared = self._prepare(df, symbol, start, end)

                # Validate cache coverage: if the prepared data covers far
                # fewer trading days than expected, the cache is likely
                # polluted (e.g. by old test mock data).  Discard it and
                # re-fetch from the source.
                if len(trading_days) > 0:
                    coverage = len(prepared) / len(trading_days)
                    if coverage >= self._MIN_CACHE_COVERAGE:
                        return prepared
                    logger.warning(
                        f"{symbol}: cache coverage {coverage:.0%} is below "
                        f"{self._MIN_CACHE_COVERAGE:.0%} threshold "
                        f"({len(prepared)}/{len(trading_days)} trading days). "
                        f"Re-fetching from source."
                    )
                else:
                    return prepared

        logger.info(f"Fetching {symbol} from source")
        df = self._source.fetch(symbol, start, end)
        df = self._cache.merge_and_save(symbol, df)
        return self._prepare(df, symbol, start, end)

    def _prepare(self, df: pd.DataFrame, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Trim to date range, reindex to trading days, and forward-fill gaps."""
        trading_days = self._calendar.trading_days(start, end)

        # Normalize index to timezone-naive midnight timestamps before
        # reindexing, so data from any source (cache, Yahoo, CSV) matches
        # the trading calendar format consistently.
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.DatetimeIndex(df.index.date, name="Date")
        df = df[~df.index.duplicated(keep="last")]

        # Reindex to trading calendar
        df = df.reindex(trading_days)

        # Replace inf/-inf with NaN before forward-filling
        import numpy as np
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward-fill gaps up to MAX_FFILL_DAYS
        df = df.ffill(limit=MAX_FFILL_DAYS)

        # Log remaining NaN gaps
        nan_count = df["Close"].isna().sum()
        if nan_count > 0:
            logger.warning(f"{symbol}: {nan_count} trading days with missing data after forward-fill")

        # Drop rows missing Close price (essential for backtesting)
        df = df.dropna(subset=["Close"])

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
