"""Fundamental data sidecar — point-in-time fundamental data lookups (Gap 9)."""

import csv
import logging
from bisect import bisect_right
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


class FundamentalDataManager:
    """Point-in-time fundamental data lookups.

    Loads a CSV with columns: date, symbol, field, value
    ``get(symbol, field, as_of)`` returns the latest value reported
    on or before *as_of* using binary search.
    """

    def __init__(self, path: str):
        # {(symbol, field): [(date, value), ...]} sorted by date
        self._data: dict[tuple[str, str], list[tuple[date, float]]] = {}
        self._load(path)

    def _load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.warning(f"Fundamental data file not found: {path}")
            return

        raw: dict[tuple[str, str], list[tuple[date, float]]] = {}
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = date.fromisoformat(row["date"].strip())
                sym = row["symbol"].strip().upper()
                field = row["field"].strip()
                try:
                    val = float(row["value"])
                except (ValueError, TypeError):
                    continue
                raw.setdefault((sym, field), []).append((d, val))

        # Sort each series by date
        for key in raw:
            raw[key].sort(key=lambda x: x[0])
        self._data = raw

    def get(self, symbol: str, field: str, as_of: date) -> float | None:
        """Return the latest fundamental value on or before as_of."""
        key = (symbol.upper(), field)
        series = self._data.get(key)
        if not series:
            return None
        dates = [d for d, _ in series]
        idx = bisect_right(dates, as_of) - 1
        if idx < 0:
            return None
        return series[idx][1]
