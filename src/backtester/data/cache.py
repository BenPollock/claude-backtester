"""Parquet-based disk cache for OHLCV data."""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ParquetCache:
    """One Parquet file per ticker. Supports incremental updates."""

    def __init__(self, cache_dir: str):
        self._cache_dir = Path(cache_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str) -> Path:
        return self._cache_dir / f"{symbol.upper()}.parquet"

    def has(self, symbol: str) -> bool:
        return self._path(symbol).exists()

    def load(self, symbol: str) -> pd.DataFrame | None:
        path = self._path(symbol)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            df.index = pd.DatetimeIndex(df.index, name="Date")
            return df
        except Exception as e:
            logger.warning(f"Failed to read cache for {symbol}: {e}")
            return None

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        path = self._path(symbol)
        try:
            df.to_parquet(path, engine="pyarrow")
            logger.debug(f"Cached {symbol}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Failed to write cache for {symbol}: {e}")

    def date_range(self, symbol: str) -> tuple[date, date] | None:
        """Return (first_date, last_date) of cached data, or None."""
        df = self.load(symbol)
        if df is None or df.empty:
            return None
        return df.index[0].date(), df.index[-1].date()

    def merge_and_save(self, symbol: str, new_df: pd.DataFrame) -> pd.DataFrame:
        """Merge new data with existing cache, keeping the union of dates."""
        existing = self.load(symbol)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = new_df
        self.save(symbol, combined)
        return combined
