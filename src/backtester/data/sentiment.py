"""CBOE Put-Call ratio data for sentiment analysis."""

import io
import logging
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CBOEPutCallSource:
    """Download CBOE equity put-call ratio from free CSV endpoint."""

    EQUITY_PC_URL = (
        "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv"
    )

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir = (
            Path(cache_dir).expanduser() / "sentiment" if cache_dir else None
        )

    def load(self, start: date, end: date) -> pd.DataFrame:
        """Download CSV, parse, compute rolling averages.

        Output columns:
        - sentiment_pcr: daily equity put-call ratio
        - sentiment_pcr_ma10: 10-day moving average of PCR

        Falls back to cached data on download failure.
        """
        df = self._download_csv()

        if df is None or df.empty:
            # Fall back to cache
            df = self._load_cache()

        if df is None or df.empty:
            return pd.DataFrame(columns=["sentiment_pcr", "sentiment_pcr_ma10"])

        # Cache the fresh data
        if df is not None and not df.empty:
            self._save_cache(df)

        # Filter to date range
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        df = df.loc[mask]

        if df.empty:
            return pd.DataFrame(columns=["sentiment_pcr", "sentiment_pcr_ma10"])

        result = pd.DataFrame(index=df.index)
        result["sentiment_pcr"] = df["pcr"].astype(float)
        result["sentiment_pcr_ma10"] = result["sentiment_pcr"].rolling(
            window=10, min_periods=10
        ).mean()

        return result

    def _download_csv(self) -> pd.DataFrame | None:
        """Download CBOE equity put-call CSV and parse it."""
        try:
            req = urllib.request.Request(
                self.EQUITY_PC_URL,
                headers={"User-Agent": "claude-backtester/0.1"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")

            # CBOE CSV has 2 header lines (disclaimer + product) before columns
            df = pd.read_csv(io.StringIO(raw), skiprows=2)

            # Normalize column names — CBOE uses various formats
            col_map = {}
            for col in df.columns:
                lower = col.strip().lower()
                if "date" in lower:
                    col_map[col] = "date"
                elif "p/c" in lower or "put/call" in lower or lower == "ratio":
                    col_map[col] = "pcr"
                elif "call" in lower:
                    col_map[col] = "calls"
                elif "put" in lower:
                    col_map[col] = "puts"
                elif "total" in lower:
                    col_map[col] = "total"

            df = df.rename(columns=col_map)

            if "date" not in df.columns:
                logger.warning("CBOE CSV missing date column")
                return None

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.set_index("date")
            df.index = pd.DatetimeIndex(df.index.date, name="Date")
            df = df.sort_index()

            # Ensure pcr column exists
            if "pcr" not in df.columns:
                if "puts" in df.columns and "calls" in df.columns:
                    calls = pd.to_numeric(df["calls"], errors="coerce").replace(
                        0, float("nan")
                    )
                    df["pcr"] = pd.to_numeric(df["puts"], errors="coerce") / calls
                else:
                    logger.warning("CBOE CSV missing ratio and puts/calls columns")
                    return None

            df["pcr"] = pd.to_numeric(df["pcr"], errors="coerce")

            return df

        except Exception:
            logger.warning("Failed to download CBOE put-call ratio data")
            return None

    def _load_cache(self) -> pd.DataFrame | None:
        """Load cached put-call data."""
        if not self._cache_dir:
            return None
        cache_path = self._cache_dir / "equity_pcr.parquet"
        if not cache_path.exists():
            return None
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            logger.warning("Failed to read CBOE cache")
            return None

    def _save_cache(self, df: pd.DataFrame) -> None:
        """Save put-call data to cache."""
        if not self._cache_dir:
            return
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self._cache_dir / "equity_pcr.parquet")
        except Exception:
            logger.warning("Failed to write CBOE cache")
