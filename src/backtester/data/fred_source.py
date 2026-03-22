"""FRED (Federal Reserve Economic Data) integration for macro regime and yield curve data."""

import logging
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from fredapi import Fred
except ImportError:
    Fred = None  # type: ignore[assignment,misc]


class FredDataSource:
    """Fetch macro data from FRED API. Optional dependency (like edgartools)."""

    MACRO_SERIES = {
        "fred_yield_spread_10y2y": "T10Y2Y",
        "fred_yield_spread_10y3m": "T10Y3M",
        "fred_credit_spread_hy": "BAMLH0A0HYM2",
        "fred_credit_spread_baa_aaa": None,  # computed: DBAA - DAAA
        "fred_lei": "USSLIND",
        "fred_claims": "ICSA",
    }

    # Internal series needed for BAA-AAA spread computation
    _INTERNAL_SERIES = {
        "DBAA": "DBAA",
        "DAAA": "DAAA",
    }

    YIELD_SERIES = {
        "yield_3m": "DGS3MO",
        "yield_2y": "DGS2",
        "yield_5y": "DGS5",
        "yield_10y": "DGS10",
        "yield_30y": "DGS30",
        "yield_breakeven_5y": "T5YIE",
        "yield_breakeven_10y": "T10YIE",
        "yield_fed_funds": "DFF",
    }

    def __init__(self, api_key: str | None = None, cache_dir: str | None = None):
        if Fred is None:
            raise ImportError(
                "fredapi is required for FRED data. "
                "Install it with: pip install -e '.[fred]'"
            )
        key = api_key or os.environ.get("FRED_API_KEY")
        if not key:
            raise ValueError(
                "FRED API key required. Pass api_key or set FRED_API_KEY env var."
            )
        self._fred = Fred(api_key=key)
        self._cache_dir = Path(cache_dir).expanduser() / "fred" if cache_dir else None

    def load_macro_regime(self, start: date, end: date) -> pd.DataFrame:
        """Load macro series, compute composite regime score.

        Fetches: T10Y2Y, T10Y3M, BAMLH0A0HYM2, DBAA-DAAA spread, USSLIND, ICSA

        Computes fred_macro_regime = count of bullish conditions / total conditions.
        Bullish conditions:
        - yield_spread_10y2y > 0 (positive yield curve)
        - credit_spread_hy < rolling median (credit tightening)
        - LEI rising (lei > lei.shift(21))
        - claims falling (claims < claims.shift(21))

        Returns DataFrame with fred_ prefixed columns + fred_macro_regime (0.0-1.0).
        """
        frames = {}

        # Fetch direct macro series
        for col_name, series_id in self.MACRO_SERIES.items():
            if series_id is not None:
                data = self._fetch_series(series_id, start, end)
                if not data.empty:
                    frames[col_name] = data

        # Fetch and compute BAA-AAA credit spread
        dbaa = self._fetch_series("DBAA", start, end)
        daaa = self._fetch_series("DAAA", start, end)
        if not dbaa.empty and not daaa.empty:
            spread = dbaa - daaa
            spread = spread.dropna()
            if not spread.empty:
                frames["fred_credit_spread_baa_aaa"] = spread

        if not frames:
            return pd.DataFrame()

        combined = pd.DataFrame(frames)
        combined = combined.sort_index()
        combined = combined.ffill()

        # Compute regime score from available conditions
        conditions = pd.DataFrame(index=combined.index)

        if "fred_yield_spread_10y2y" in combined.columns:
            conditions["yield_curve"] = (
                combined["fred_yield_spread_10y2y"] > 0
            ).astype(float)

        if "fred_credit_spread_hy" in combined.columns:
            median_hy = combined["fred_credit_spread_hy"].rolling(
                window=252, min_periods=21
            ).median()
            conditions["credit"] = (
                combined["fred_credit_spread_hy"] < median_hy
            ).astype(float)

        if "fred_lei" in combined.columns:
            conditions["lei"] = (
                combined["fred_lei"] > combined["fred_lei"].shift(21)
            ).astype(float)

        if "fred_claims" in combined.columns:
            conditions["claims"] = (
                combined["fred_claims"] < combined["fred_claims"].shift(21)
            ).astype(float)

        if not conditions.empty and len(conditions.columns) > 0:
            # Count of bullish / total non-NaN conditions
            bullish_count = conditions.sum(axis=1)
            total_count = conditions.notna().sum(axis=1).replace(0, float("nan"))
            combined["fred_macro_regime"] = bullish_count / total_count
        else:
            combined["fred_macro_regime"] = float("nan")

        return combined

    def load_yield_curve(self, start: date, end: date) -> pd.DataFrame:
        """Load full Treasury curve + breakevens + Fed Funds.

        Computes:
        - yield_spread = yield_10y - yield_2y
        - yield_real_10y = yield_10y - yield_breakeven_10y

        Returns DataFrame with yield_ prefixed columns.
        """
        frames = {}

        for col_name, series_id in self.YIELD_SERIES.items():
            data = self._fetch_series(series_id, start, end)
            if not data.empty:
                frames[col_name] = data

        if not frames:
            return pd.DataFrame()

        combined = pd.DataFrame(frames)
        combined = combined.sort_index()
        combined = combined.ffill()

        # Compute derived columns
        if "yield_10y" in combined.columns and "yield_2y" in combined.columns:
            combined["yield_spread"] = (
                combined["yield_10y"] - combined["yield_2y"]
            )

        if (
            "yield_10y" in combined.columns
            and "yield_breakeven_10y" in combined.columns
        ):
            combined["yield_real_10y"] = (
                combined["yield_10y"] - combined["yield_breakeven_10y"]
            )

        return combined

    def _fetch_series(self, series_id: str, start: date, end: date) -> pd.Series:
        """Fetch a single FRED series with caching."""
        # Try cache first
        if self._cache_dir:
            cache_path = self._cache_dir / f"{series_id}.parquet"
            if cache_path.exists():
                try:
                    df = pd.read_parquet(cache_path)
                    if not df.empty:
                        series = df.iloc[:, 0]
                        # Filter to requested date range
                        mask = (series.index >= pd.Timestamp(start)) & (
                            series.index <= pd.Timestamp(end)
                        )
                        return series.loc[mask]
                except Exception:
                    logger.warning("Failed to read FRED cache for %s", series_id)

        # Fetch from FRED
        try:
            data = self._fred.get_series(
                series_id, observation_start=start, observation_end=end
            )
            if data is not None and not data.empty:
                # Cache the data
                if self._cache_dir:
                    self._cache_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({series_id: data}).to_parquet(
                        self._cache_dir / f"{series_id}.parquet"
                    )
                return data
        except Exception:
            logger.warning("Failed to fetch FRED series %s", series_id)

        return pd.Series(dtype=float)
