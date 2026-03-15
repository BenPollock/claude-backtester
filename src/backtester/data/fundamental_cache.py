"""Unified Parquet cache for EDGAR data (financials, insider, institutional, events)."""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_VALID_DATA_TYPES = {"financials", "insider", "institutional", "events"}


class EdgarCache:
    """Parquet-based cache for EDGAR data, parameterized by data_type.

    Cache layout::

        {cache_dir}/edgar/financials/AAPL.parquet
        {cache_dir}/edgar/insider/AAPL.parquet
        {cache_dir}/edgar/institutional/AAPL.parquet
        {cache_dir}/edgar/events/AAPL.parquet
    """

    def __init__(self, cache_dir: str, data_type: str) -> None:
        if data_type not in _VALID_DATA_TYPES:
            raise ValueError(
                f"data_type must be one of {_VALID_DATA_TYPES}, got {data_type!r}"
            )
        self.cache_dir = Path(cache_dir)
        self.data_type = data_type
        self._dir = self.cache_dir / "edgar" / data_type
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str) -> Path:
        """Return the Parquet file path for the given symbol."""
        return self._dir / f"{symbol.upper()}.parquet"

    def has(self, symbol: str) -> bool:
        """Return True if cached data exists for *symbol*."""
        return self._path(symbol).exists()

    def load(self, symbol: str) -> pd.DataFrame | None:
        """Load cached data for *symbol*, or None if not cached."""
        path = self._path(symbol)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            logger.warning("Failed to read cache file %s", path)
            return None

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        """Save *df* to the cache for *symbol*."""
        path = self._path(symbol)
        try:
            df.to_parquet(path, index=False)
        except Exception:
            logger.warning("Failed to write cache file %s", path)

    def date_range(self, symbol: str) -> tuple[date, date] | None:
        """Return (min_date, max_date) of filed_date in the cached data.

        Returns None if no cached data or no filed_date column.
        """
        df = self.load(symbol)
        if df is None or df.empty:
            return None

        date_col = "filed_date"
        if date_col not in df.columns:
            return None

        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if dates.empty:
            return None

        return dates.min().date(), dates.max().date()

    def merge_and_save(self, symbol: str, new_df: pd.DataFrame) -> None:
        """Merge *new_df* with existing cache, dedup, sort, and save.

        Deduplication keeps the last occurrence (latest filing) for rows
        that share the same key columns. The key columns depend on data_type:
        - financials: (metric, period_end)
        - insider: (filed_date, insider_name, shares)
        - institutional: (report_date,)
        - events: (filed_date, items)
        """
        existing = self.load(symbol)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df.copy()

        if combined.empty:
            self.save(symbol, combined)
            return

        # Dedup based on data type
        dedup_keys = self._dedup_keys()
        valid_keys = [k for k in dedup_keys if k in combined.columns]
        if valid_keys:
            combined.drop_duplicates(subset=valid_keys, keep="last", inplace=True)

        # Sort by filed_date if available
        if "filed_date" in combined.columns:
            combined.sort_values("filed_date", inplace=True)

        combined.reset_index(drop=True, inplace=True)
        self.save(symbol, combined)

    def _dedup_keys(self) -> list[str]:
        """Return dedup key columns for this data type."""
        if self.data_type == "financials":
            return ["metric", "period_end"]
        elif self.data_type == "insider":
            return ["filed_date", "insider_name", "shares"]
        elif self.data_type == "institutional":
            return ["report_date"]
        elif self.data_type == "events":
            return ["filed_date", "event_date"]
        return []
