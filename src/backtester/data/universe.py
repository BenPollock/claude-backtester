"""Ticker universe provider — auto-populates ticker lists from index constituents."""

import csv
import json
import logging
import time
import urllib.request
from bisect import bisect_right
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class HistoricalUniverse:
    """Point-in-time universe membership from a CSV file.

    The CSV must have columns ``date`` (YYYY-MM-DD) and ``symbol``.
    Each row records a snapshot of which symbols were in the universe on
    that date.  ``members_on(query_date)`` returns the membership set
    from the latest snapshot on or before *query_date* using binary
    search for O(log n) lookups.
    """

    def __init__(self, path: str):
        self._snapshots: list[date] = []
        self._members: dict[date, set[str]] = {}
        self._all_symbols: set[str] = set()
        self._load(path)

    def _load(self, path: str) -> None:
        raw: dict[date, set[str]] = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = date.fromisoformat(row["date"].strip())
                sym = row["symbol"].strip().upper()
                raw.setdefault(d, set()).add(sym)
                self._all_symbols.add(sym)
        self._snapshots = sorted(raw.keys())
        self._members = raw

    def members_on(self, query_date: date) -> set[str] | None:
        """Return the universe membership set for the latest snapshot <= query_date."""
        if not self._snapshots:
            return None
        idx = bisect_right(self._snapshots, query_date) - 1
        if idx < 0:
            return None
        return self._members[self._snapshots[idx]]

    @property
    def all_symbols(self) -> set[str]:
        """Union of all symbols that appear in any snapshot."""
        return set(self._all_symbols)

_CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # 7 days


class UniverseProvider:
    """Fetches and caches index constituent ticker lists."""

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir = Path(cache_dir or "~/.backtester/cache/universe").expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get_tickers(self, market: str = "us_ca", universe: str = "index") -> list[str]:
        """Return ticker list for the given market/universe combination."""
        if universe == "all":
            logger.warning("universe='all' not yet supported — falling back to 'index'")

        fetchers: dict[str, list[tuple[str, callable]]] = {
            "us": [("us_index", self._fetch_sp500)],
            "ca": [("ca_index", self._fetch_tsx)],
            "us_ca": [("us_index", self._fetch_sp500), ("ca_index", self._fetch_tsx)],
        }

        if market not in fetchers:
            raise ValueError(f"Unknown market '{market}'. Choose from: {sorted(fetchers)}")

        tickers: list[str] = []
        for cache_key, fetch_fn in fetchers[market]:
            tickers.extend(self._get_or_fetch(cache_key, fetch_fn))

        return sorted(set(tickers))

    # -- caching -------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _load_cached(self, key: str) -> list[str] | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > _CACHE_MAX_AGE_SECONDS:
            return None  # stale
        return json.loads(path.read_text())

    def _load_stale(self, key: str) -> list[str] | None:
        """Load cache regardless of age (fallback on fetch failure)."""
        path = self._cache_path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _save_cache(self, key: str, tickers: list[str]) -> None:
        self._cache_path(key).write_text(json.dumps(tickers))

    def _get_or_fetch(self, key: str, fetch_fn: callable) -> list[str]:
        cached = self._load_cached(key)
        if cached is not None:
            logger.info("Universe '%s': loaded %d tickers from cache", key, len(cached))
            return cached

        try:
            tickers = fetch_fn()
            self._save_cache(key, tickers)
            logger.info("Universe '%s': fetched %d tickers", key, len(tickers))
            return tickers
        except Exception as exc:
            stale = self._load_stale(key)
            if stale is not None:
                logger.warning("Fetch failed for '%s' (%s) — using stale cache (%d tickers)",
                               key, exc, len(stale))
                return stale
            raise RuntimeError(
                f"Failed to fetch universe '{key}' and no cached data available: {exc}"
            ) from exc

    # -- fetchers ------------------------------------------------------------

    @staticmethod
    def _read_html_with_ua(url: str) -> list[pd.DataFrame]:
        """Fetch HTML with a browser-like User-Agent, then parse tables."""
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; claude-backtester/0.1)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode(resp.headers.get_content_charset() or "utf-8")
        from io import StringIO
        return pd.read_html(StringIO(html))

    def _fetch_sp500(self) -> list[str]:
        """Scrape S&P 500 constituents from Wikipedia."""
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = self._read_html_with_ua(url)
        df = tables[0]
        symbols = df["Symbol"].str.strip().str.replace(".", "-", regex=False).tolist()
        return sorted(symbols)

    def _fetch_tsx(self) -> list[str]:
        """Scrape S&P/TSX Composite constituents from Wikipedia."""
        url = "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
        tables = self._read_html_with_ua(url)
        df = tables[0]
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        symbols = df[col].str.strip().tolist()
        # Append .TO suffix for yfinance
        symbols = [f"{s}.TO" if not s.endswith(".TO") else s for s in symbols]
        return sorted(symbols)
