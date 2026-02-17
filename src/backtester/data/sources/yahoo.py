"""Yahoo Finance data source via yfinance with retry/backoff."""

import logging
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from backtester.data.sources.base import DataSource

logger = logging.getLogger(__name__)

# Standard column names we expect
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class YahooDataSource(DataSource):
    """Fetches split-adjusted OHLCV data from Yahoo Finance."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, inter_request_delay: float = 0.5):
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._inter_request_delay = inter_request_delay
        self._last_request_time = 0.0

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Fetch OHLCV from yfinance with exponential backoff on failure."""
        self._throttle()

        # yfinance end date is exclusive, add 1 day
        end_str = (end + timedelta(days=1)).isoformat()
        start_str = start.isoformat()

        for attempt in range(self._max_retries):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_str, end=end_str, auto_adjust=True)

                if df.empty:
                    raise ValueError(f"No data returned for {symbol}")

                df = self._normalize(df, symbol)
                return df

            except Exception as e:
                if attempt < self._max_retries - 1:
                    delay = self._base_delay * (2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{self._max_retries} for {symbol}: {e}. "
                                   f"Waiting {delay:.1f}s")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to fetch {symbol} after {self._max_retries} attempts: {e}")
                    raise

        # Should not reach here, but satisfy type checker
        raise RuntimeError(f"Failed to fetch {symbol}")

    def _throttle(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._inter_request_delay:
            time.sleep(self._inter_request_delay - elapsed)
        self._last_request_time = time.time()

    def _normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize DataFrame to standard format."""
        # Drop any extra columns (Dividends, Stock Splits)
        cols_present = [c for c in REQUIRED_COLUMNS if c in df.columns]
        if len(cols_present) < len(REQUIRED_COLUMNS):
            missing = set(REQUIRED_COLUMNS) - set(cols_present)
            raise ValueError(f"Missing columns for {symbol}: {missing}")

        df = df[REQUIRED_COLUMNS].copy()

        # Ensure index is timezone-naive dates
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.DatetimeIndex(df.index.date, name="Date")

        # Remove duplicate dates (keep last)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        return df
