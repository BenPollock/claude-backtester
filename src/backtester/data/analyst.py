"""Analyst earnings revision data via yfinance."""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class AnalystRevisionSource:
    """Analyst earnings revisions via yfinance (per-symbol).

    LIMITATION: yfinance provides current snapshot only, not historical
    point-in-time estimates. For backtesting, columns will be NaN for
    historical dates. Most useful for forward-looking / paper trading.
    """

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir = (
            Path(cache_dir).expanduser() / "analyst" if cache_dir else None
        )

    def fetch(
        self,
        symbol: str,
        daily_index: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """Get current analyst revision data from yfinance.

        Output columns:
        - analyst_rev_up_7d: number of upward revisions in last 7 days
        - analyst_rev_down_7d: number of downward revisions in last 7 days
        - analyst_rev_breadth: (up - down) / (up + down), range -1 to 1

        If daily_index is provided, creates a DataFrame indexed to those dates
        with the current snapshot values on the last date only (all other dates NaN).
        This is honest about the point-in-time limitation.
        """
        symbol = symbol.upper()

        # Try cache first
        cached = self._load_cache(symbol)
        if cached is not None:
            return self._align_to_index(cached, daily_index)

        # Fetch from yfinance
        snapshot = self._fetch_from_yfinance(symbol)

        if snapshot is not None:
            self._save_cache(symbol, snapshot)
            return self._align_to_index(snapshot, daily_index)

        # Return empty frame with correct columns
        return self._empty_frame(daily_index)

    def _fetch_from_yfinance(self, symbol: str) -> dict | None:
        """Extract revision counts from yfinance Ticker."""
        try:
            ticker = yf.Ticker(symbol)
            # yfinance exposes earnings estimate revisions via .earnings_estimate
            # or .analyst_price_targets. The exact attribute varies by version.
            # We try multiple approaches.

            up = 0
            down = 0

            # Try get_earnings_estimate (newer yfinance versions)
            try:
                est = ticker.earnings_estimate
                if est is not None and not est.empty:
                    # Look for revision columns
                    for col in est.columns:
                        lower = col.lower()
                        if "up" in lower and "7" in lower:
                            val = est[col].iloc[0]
                            if pd.notna(val):
                                up = int(val)
                        elif "down" in lower and "7" in lower:
                            val = est[col].iloc[0]
                            if pd.notna(val):
                                down = int(val)
            except Exception:
                pass

            # Fallback: try recommendations or other endpoints
            if up == 0 and down == 0:
                try:
                    rec = ticker.recommendations
                    if rec is not None and not rec.empty:
                        # Use last 7 days of recommendations as proxy
                        recent = rec.tail(7)
                        for _, row in recent.iterrows():
                            grade = str(row.get("To Grade", "")).lower()
                            if any(
                                w in grade
                                for w in ["buy", "outperform", "overweight"]
                            ):
                                up += 1
                            elif any(
                                w in grade
                                for w in ["sell", "underperform", "underweight"]
                            ):
                                down += 1
                except Exception:
                    pass

            return {"up": up, "down": down}

        except Exception:
            logger.warning("Failed to fetch analyst data for %s", symbol)
            return None

    def _align_to_index(
        self, snapshot: dict, daily_index: pd.DatetimeIndex | None
    ) -> pd.DataFrame:
        """Place snapshot values on last date of index, NaN everywhere else."""
        up = snapshot.get("up", 0)
        down = snapshot.get("down", 0)
        total = up + down
        breadth = (up - down) / total if total > 0 else float("nan")

        if daily_index is not None and len(daily_index) > 0:
            result = pd.DataFrame(
                {
                    "analyst_rev_up_7d": float("nan"),
                    "analyst_rev_down_7d": float("nan"),
                    "analyst_rev_breadth": float("nan"),
                },
                index=daily_index,
            )
            # Set snapshot values on the last date only
            result.loc[daily_index[-1], "analyst_rev_up_7d"] = float(up)
            result.loc[daily_index[-1], "analyst_rev_down_7d"] = float(down)
            result.loc[daily_index[-1], "analyst_rev_breadth"] = breadth
            return result

        # No index — return single-row snapshot
        return pd.DataFrame(
            {
                "analyst_rev_up_7d": [float(up)],
                "analyst_rev_down_7d": [float(down)],
                "analyst_rev_breadth": [breadth],
            }
        )

    def _empty_frame(
        self, daily_index: pd.DatetimeIndex | None
    ) -> pd.DataFrame:
        """Return empty frame with correct columns."""
        cols = ["analyst_rev_up_7d", "analyst_rev_down_7d", "analyst_rev_breadth"]
        if daily_index is not None and len(daily_index) > 0:
            return pd.DataFrame(
                {c: float("nan") for c in cols},
                index=daily_index,
            )
        return pd.DataFrame(columns=cols)

    def _load_cache(self, symbol: str) -> dict | None:
        """Load cached snapshot for symbol."""
        if not self._cache_dir:
            return None
        cache_path = self._cache_dir / f"{symbol}.parquet"
        if not cache_path.exists():
            return None
        try:
            df = pd.read_parquet(cache_path)
            if df.empty:
                return None
            row = df.iloc[0]
            return {
                "up": int(row.get("up", 0)),
                "down": int(row.get("down", 0)),
            }
        except Exception:
            logger.warning("Failed to read analyst cache for %s", symbol)
            return None

    def _save_cache(self, symbol: str, snapshot: dict) -> None:
        """Save snapshot to cache."""
        if not self._cache_dir:
            return
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame([snapshot])
            df.to_parquet(self._cache_dir / f"{symbol}.parquet")
        except Exception:
            logger.warning("Failed to write analyst cache for %s", symbol)
