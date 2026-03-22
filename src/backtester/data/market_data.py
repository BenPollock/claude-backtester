"""VIX term structure and cross-asset intermarket data via yfinance."""

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketDataManager:
    """Loads and caches VIX term structure + cross-asset intermarket data via yfinance."""

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir else None

    def _load_or_fetch(self, ticker: str, start: date, end: date) -> pd.Series:
        """Load a single ticker's Close prices from cache or yfinance."""
        safe_name = (
            ticker.replace("=", "_")
            .replace(".", "_")
            .replace("-", "_")
            .replace("^", "")
        )
        cache_path = (
            self._cache_dir / "market" / f"{safe_name}.parquet"
            if self._cache_dir
            else None
        )

        # Try cache
        if cache_path and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                if not df.empty:
                    return df["Close"]
            except Exception:
                logger.warning("Failed to read cache for %s", ticker)

        # Fetch from yfinance
        try:
            data = yf.download(ticker, start=str(start), end=str(end), progress=False)
            if data is not None and not data.empty:
                close = data["Close"]
                # Save to cache
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    data[["Close"]].to_parquet(cache_path)
                return close
        except Exception:
            logger.warning("Failed to fetch %s from yfinance", ticker)

        return pd.Series(dtype=float)

    def load_vix_data(self, start: date, end: date) -> pd.DataFrame:
        """Load ^VIX, ^VIX3M via yfinance. Compute term structure signals.

        Output columns (all date-indexed):
        - vix_close: VIX closing value
        - vix_3m: VIX3M closing value
        - vix_ratio: VIX / VIX3M (>1.0 = backwardation/stress, <1.0 = contango)
        - vix_regime: "contango" or "backwardation"

        Cache to {cache_dir}/market/VIX.parquet and VIX3M.parquet.
        """
        vix = self._load_or_fetch("^VIX", start, end)
        vix3m = self._load_or_fetch("^VIX3M", start, end)

        if vix.empty and vix3m.empty:
            return pd.DataFrame(
                columns=["vix_close", "vix_3m", "vix_ratio", "vix_regime"]
            )

        # Align on common dates via outer join + forward-fill
        combined = pd.DataFrame({"vix_close": vix, "vix_3m": vix3m})
        combined = combined.sort_index()
        combined = combined.ffill()

        # Compute ratio (guard division by zero — 0 denominator → NaN)
        combined["vix_ratio"] = combined["vix_close"] / combined["vix_3m"].replace(
            0, float("nan")
        )

        # Classify regime
        combined["vix_regime"] = combined["vix_ratio"].apply(
            lambda r: "backwardation" if r > 1.0 else "contango"
            if pd.notna(r)
            else None
        )

        return combined

    def load_intermarket_data(self, start: date, end: date) -> pd.DataFrame:
        """Load copper, gold, dollar index via yfinance. Compute intermarket signals.

        Output columns:
        - intermarket_cu_au_ratio: copper / gold price ratio
        - intermarket_cu_au_momentum: ratio.pct_change(63) (quarterly momentum)
        - intermarket_dollar: DX-Y.NYB closing value

        Tickers: HG=F (copper futures), GC=F (gold futures), DX-Y.NYB (dollar index)
        Cache to {cache_dir}/market/{ticker}.parquet.
        """
        copper = self._load_or_fetch("HG=F", start, end)
        gold = self._load_or_fetch("GC=F", start, end)
        dollar = self._load_or_fetch("DX-Y.NYB", start, end)

        if copper.empty and gold.empty and dollar.empty:
            return pd.DataFrame(
                columns=[
                    "intermarket_cu_au_ratio",
                    "intermarket_cu_au_momentum",
                    "intermarket_dollar",
                ]
            )

        # Align via outer join + forward-fill
        combined = pd.DataFrame(
            {"_copper": copper, "_gold": gold, "intermarket_dollar": dollar}
        )
        combined = combined.sort_index()
        combined = combined.ffill()

        # Compute copper/gold ratio (guard division by zero)
        combined["intermarket_cu_au_ratio"] = combined["_copper"] / combined[
            "_gold"
        ].replace(0, float("nan"))

        # Quarterly momentum of the ratio
        combined["intermarket_cu_au_momentum"] = combined[
            "intermarket_cu_au_ratio"
        ].pct_change(63)

        # Drop internal columns
        combined = combined.drop(columns=["_copper", "_gold"])

        return combined
