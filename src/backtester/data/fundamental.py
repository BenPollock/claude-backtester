"""Fundamental data manager — CSV and SEC EDGAR data sources.

Backward-compatible: ``FundamentalDataManager`` is preserved as an alias.
New ``EdgarDataManager`` supports CSV fallback plus EDGAR integration.
"""

import csv
import logging
from bisect import bisect_right
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Original CSV-only manager (preserved for backward compatibility)
# ---------------------------------------------------------------------------


class _CSVFundamentalData:
    """Point-in-time fundamental data lookups from CSV.

    CSV columns: date, symbol, field, value
    ``get(symbol, field, as_of)`` returns the latest value reported
    on or before *as_of* using binary search.
    """

    def __init__(self, path: str) -> None:
        self._data: dict[tuple[str, str], list[tuple[date, float]]] = {}
        self._load(path)

    def _load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.warning("Fundamental data file not found: %s", path)
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

        for key in raw:
            raw[key].sort(key=lambda x: x[0])
        self._data = raw

    def get(self, symbol: str, field: str, as_of: date) -> float | None:
        key = (symbol.upper(), field)
        series = self._data.get(key)
        if not series:
            return None
        dates = [d for d, _ in series]
        idx = bisect_right(dates, as_of) - 1
        if idx < 0:
            return None
        return series[idx][1]


# ---------------------------------------------------------------------------
# Flow vs stock metric classification
# ---------------------------------------------------------------------------

# Flow metrics need TTM (trailing twelve months = sum of 4 quarters)
_FLOW_METRICS = {
    "revenue",
    "net_income",
    "eps_diluted",
    "operating_income",
    "gross_profit",
    "operating_cf",
    "capex",
    "dividends_paid",
    "research_dev",
    "ebitda",
    "stock_repurchased",
    "stock_issued_proceeds",
    "stock_comp",
}

# Stock/balance-sheet metrics use latest value
_STOCK_METRICS = {
    "total_assets",
    "total_debt",
    "current_assets",
    "current_liabilities",
    "equity",
    "shares_outstanding",
    "retained_earnings",
    "total_liabilities",
    "dividends_per_share",
}


# ---------------------------------------------------------------------------
# EdgarDataManager
# ---------------------------------------------------------------------------


class EdgarDataManager:
    """Unified fundamental data manager supporting CSV and EDGAR sources.

    When *csv_path* is provided and *use_edgar* is False, behaves exactly
    like the original ``FundamentalDataManager``.  When *use_edgar* is True,
    fetches data from SEC EDGAR via ``edgartools``.

    Key method: ``merge_all_onto_daily(symbol, daily_df)`` merges all
    available fundamental data onto a daily price DataFrame using
    point-in-time (filed_date) alignment to prevent lookahead bias.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        sources: dict | None = None,
        use_edgar: bool = False,
        csv_path: str | None = None,
        edgar_user_agent: str | None = None,
        enable_financials: bool = True,
        enable_insider: bool = True,
        enable_institutional: bool = False,
        enable_events: bool = True,
        edgar_max_filings: int = 50,
    ) -> None:
        self._csv: _CSVFundamentalData | None = None
        self._use_edgar = use_edgar
        self._cache_dir = cache_dir
        self._user_agent = edgar_user_agent

        self._enable_financials = enable_financials
        self._enable_insider = enable_insider
        self._enable_institutional = enable_institutional
        self._enable_events = enable_events
        self._max_filings = edgar_max_filings

        # EDGAR source instances (lazy-initialized)
        self._financials_source = None
        self._insider_source = None
        self._institutional_source = None
        self._events_source = None

        # Caches (lazy-initialized)
        self._financials_cache = None
        self._insider_cache = None
        self._institutional_cache = None
        self._events_cache = None

        # CSV fallback
        if csv_path:
            self._csv = _CSVFundamentalData(csv_path)

        # Initialize EDGAR sources if enabled
        if use_edgar:
            self._init_edgar_sources()

        # Allow injection of pre-built sources (for testing)
        if sources:
            for key, src in sources.items():
                setattr(self, f"_{key}_source", src)

    def _init_edgar_sources(self) -> None:
        """Lazily initialize EDGAR source classes and caches."""
        if not self._user_agent:
            logger.warning(
                "edgar_user_agent is required for EDGAR data. "
                "Set it to 'YourName your@email.com'."
            )
            return

        if self._enable_financials:
            try:
                from backtester.data.edgar_source import EdgarFundamentalSource

                self._financials_source = EdgarFundamentalSource(self._user_agent)
            except ImportError:
                logger.info("edgartools not installed; financials disabled")

        if self._enable_insider:
            try:
                from backtester.data.edgar_insider import EdgarInsiderSource

                self._insider_source = EdgarInsiderSource(
                    self._user_agent, max_filings=self._max_filings
                )
            except ImportError:
                logger.info("edgartools not installed; insider data disabled")

        if self._enable_institutional:
            try:
                from backtester.data.edgar_institutional import (
                    EdgarInstitutionalSource,
                )

                self._institutional_source = EdgarInstitutionalSource(
                    self._user_agent, max_filings=self._max_filings
                )
            except ImportError:
                logger.info(
                    "edgartools not installed; institutional data disabled"
                )

        if self._enable_events:
            try:
                from backtester.data.edgar_events import EdgarEventSource

                self._events_source = EdgarEventSource(
                    self._user_agent, max_filings=self._max_filings
                )
            except ImportError:
                logger.info("edgartools not installed; events data disabled")

        # Initialize caches
        if self._cache_dir:
            from backtester.data.fundamental_cache import EdgarCache

            if self._enable_financials:
                self._financials_cache = EdgarCache(self._cache_dir, "financials")
            if self._enable_insider:
                self._insider_cache = EdgarCache(self._cache_dir, "insider")
            if self._enable_institutional:
                self._institutional_cache = EdgarCache(
                    self._cache_dir, "institutional"
                )
            if self._enable_events:
                self._events_cache = EdgarCache(self._cache_dir, "events")

    # ------------------------------------------------------------------
    # Data loading methods
    # ------------------------------------------------------------------

    def load_financials(self, symbol: str) -> pd.DataFrame:
        """Load financial statement data for *symbol*.

        Checks cache first, then fetches from EDGAR if available.
        """
        return self._load_data(
            symbol,
            self._financials_cache,
            self._financials_source,
            "financials",
        )

    def load_insider(self, symbol: str) -> pd.DataFrame:
        """Load Form 4 insider trading data for *symbol*."""
        return self._load_data(
            symbol,
            self._insider_cache,
            self._insider_source,
            "insider",
        )

    def load_institutional(self, symbol: str) -> pd.DataFrame:
        """Load 13F institutional holdings data for *symbol*."""
        return self._load_data(
            symbol,
            self._institutional_cache,
            self._institutional_source,
            "institutional",
        )

    def load_events(self, symbol: str) -> pd.DataFrame:
        """Load 8-K material event data for *symbol*."""
        return self._load_data(
            symbol,
            self._events_cache,
            self._events_source,
            "events",
        )

    def _load_data(
        self,
        symbol: str,
        cache: object | None,
        source: object | None,
        data_type: str,
    ) -> pd.DataFrame:
        """Generic load: try cache, then fetch from source."""
        # Try cache first
        if cache is not None:
            cached = cache.load(symbol)  # type: ignore[union-attr]
            if cached is not None:
                return cached

        # Fetch from source
        if source is not None:
            try:
                df = source.fetch(symbol)  # type: ignore[union-attr]
                if df is not None and not df.empty:
                    if cache is not None:
                        cache.merge_and_save(symbol, df)  # type: ignore[union-attr]
                    return df
            except Exception:
                logger.warning("Failed to fetch %s data for %s", data_type, symbol)

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # merge_all_onto_daily — THE KEY METHOD
    # ------------------------------------------------------------------

    def merge_all_onto_daily(
        self, symbol: str, daily_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge all EDGAR data onto a daily price DataFrame.

        Uses ``filed_date`` for point-in-time alignment (no lookahead).
        Never mutates the input DataFrame.

        Returns a new DataFrame with ``fund_``, ``insider_``, ``inst_``,
        and ``event_`` prefixed columns.
        """
        result = daily_df.copy()

        # Ensure the index is a DatetimeIndex for merge_asof
        if not isinstance(result.index, pd.DatetimeIndex):
            return result

        # 1. Financial statement data
        if self._enable_financials:
            result = self._merge_financials(symbol, result)

        # 2. Insider trading data
        if self._enable_insider:
            result = self._merge_insider(symbol, result)

        # 3. Institutional holdings data
        if self._enable_institutional:
            result = self._merge_institutional(symbol, result)

        # 4. Material events data
        if self._enable_events:
            result = self._merge_events(symbol, result)

        return result

    # ------------------------------------------------------------------
    # Financials merge
    # ------------------------------------------------------------------

    def _merge_financials(
        self, symbol: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge financial statement data onto daily DataFrame."""
        fin = self.load_financials(symbol)
        if fin.empty:
            return df

        # Ensure filed_date is datetime
        fin = fin.copy()
        fin["filed_date"] = pd.to_datetime(fin["filed_date"], errors="coerce")
        fin["period_end"] = pd.to_datetime(fin["period_end"], errors="coerce")
        fin.dropna(subset=["filed_date"], inplace=True)

        if fin.empty:
            return df

        # Pivot: one row per filed_date with metric columns
        pivoted = self._pivot_financials(fin)
        if pivoted.empty:
            return df

        # Compute TTM for flow metrics
        pivoted = self._compute_ttm(pivoted, fin)

        # Compute derived ratios (these need Close price, done after merge)
        # Sort for merge_asof
        pivoted.sort_values("filed_date", inplace=True)
        pivoted.reset_index(drop=True, inplace=True)

        # merge_asof: align on filed_date (backward)
        df_reset = df.reset_index()
        df_reset.rename(columns={df_reset.columns[0]: "_date"}, inplace=True)
        df_reset["_date"] = pd.to_datetime(df_reset["_date"])
        df_reset.sort_values("_date", inplace=True)

        merged = pd.merge_asof(
            df_reset,
            pivoted,
            left_on="_date",
            right_on="filed_date",
            direction="backward",
        )

        merged.set_index("_date", inplace=True)
        merged.index.name = df.index.name

        # Drop the filed_date column from merge
        if "filed_date" in merged.columns:
            merged.drop(columns=["filed_date"], inplace=True)

        # Compute price-dependent ratios
        merged = self._compute_price_ratios(merged)
        # Compute margin and growth ratios
        merged = self._compute_fundamental_ratios(merged)
        # Compute Piotroski F-Score
        merged = self._compute_piotroski_f(merged)
        # Compute Altman Z-Score (needs Close for market cap)
        merged = self._compute_altman_z(merged)
        # Compute Buyback / Shareholder Yield
        merged = self._compute_shareholder_yield(merged)
        # Compute Dividend Growth
        merged = self._compute_dividend_growth(merged)

        return merged

    def _pivot_financials(self, fin: pd.DataFrame) -> pd.DataFrame:
        """Pivot financial data: each metric becomes a fund_ column.

        For each filed_date, take the latest value per metric.
        """
        if fin.empty:
            return pd.DataFrame()

        # Group by filed_date and take latest metric values
        result_rows: list[dict] = []
        # Sort by filed_date to process in order
        fin_sorted = fin.sort_values(["filed_date", "period_end"])

        # Build cumulative state: as of each filed_date, what are the latest values?
        state: dict[str, float] = {}
        current_dates: list[pd.Timestamp] = []
        current_filed_date = None

        for _, row in fin_sorted.iterrows():
            fd = row["filed_date"]
            if current_filed_date is not None and fd != current_filed_date:
                # Emit a row for the previous filed_date
                row_dict = {"filed_date": current_filed_date}
                row_dict.update(
                    {f"fund_{k}": v for k, v in state.items()}
                )
                result_rows.append(row_dict)

            current_filed_date = fd
            metric = row["metric"]
            state[metric] = row["value"]

        # Emit last batch
        if current_filed_date is not None:
            row_dict = {"filed_date": current_filed_date}
            row_dict.update({f"fund_{k}": v for k, v in state.items()})
            result_rows.append(row_dict)

        if not result_rows:
            return pd.DataFrame()

        pivoted = pd.DataFrame(result_rows)
        # Dedup: keep last per filed_date
        pivoted.drop_duplicates(subset=["filed_date"], keep="last", inplace=True)
        return pivoted

    def _compute_ttm(
        self, pivoted: pd.DataFrame, fin: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute trailing twelve month sums for flow metrics."""
        for metric in _FLOW_METRICS:
            col = f"fund_{metric}"
            ttm_col = f"fund_{metric}_ttm"
            if col not in pivoted.columns:
                pivoted[ttm_col] = np.nan
                continue

            # Get quarterly data for this metric sorted by period_end
            metric_data = fin[fin["metric"] == metric].copy()
            metric_data.sort_values("period_end", inplace=True)

            if metric_data.empty:
                pivoted[ttm_col] = np.nan
                continue

            # For each filed_date, compute sum of last 4 quarters
            ttm_values: dict[pd.Timestamp, float] = {}
            values_by_period = list(
                zip(metric_data["period_end"], metric_data["value"])
            )

            for fd in pivoted["filed_date"]:
                # Get the 4 most recent quarters as of this filing date
                available = [
                    (pe, v) for pe, v in values_by_period if pe <= fd
                ]
                last_4 = available[-4:] if len(available) >= 4 else available
                ttm_values[fd] = sum(v for _, v in last_4) if last_4 else np.nan

            pivoted[ttm_col] = pivoted["filed_date"].map(ttm_values)

        # Compute free cash flow
        if "fund_operating_cf" in pivoted.columns and "fund_capex" in pivoted.columns:
            pivoted["fund_free_cash_flow"] = (
                pivoted["fund_operating_cf"] - pivoted["fund_capex"]
            )
        else:
            pivoted["fund_free_cash_flow"] = np.nan

        if (
            "fund_operating_cf_ttm" in pivoted.columns
            and "fund_capex" in pivoted.columns
        ):
            # capex TTM approximation: use raw capex * 4 if no TTM
            capex_ttm_col = "fund_capex_ttm"
            if capex_ttm_col not in pivoted.columns:
                pivoted[capex_ttm_col] = np.nan
            # Compute FCF TTM from TTM components
            ocf_ttm = pivoted.get("fund_operating_cf_ttm", np.nan)
            capex_ttm = pivoted.get(capex_ttm_col, np.nan)
            pivoted["fund_free_cash_flow_ttm"] = ocf_ttm - capex_ttm
        else:
            pivoted["fund_free_cash_flow_ttm"] = np.nan

        return pivoted

    def _compute_price_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price-dependent ratios (P/E, P/B, P/S, FCF yield)."""
        close = df.get("Close")
        if close is None:
            return df

        # P/E ratio: Close / eps_diluted_ttm
        eps_ttm = df.get("fund_eps_diluted_ttm")
        if eps_ttm is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                df["fund_pe_ratio"] = np.where(
                    (eps_ttm != 0) & eps_ttm.notna(),
                    close / eps_ttm,
                    np.nan,
                )
        else:
            df["fund_pe_ratio"] = np.nan

        # P/B ratio: Close / (equity / shares_outstanding)
        equity = df.get("fund_equity")
        shares = df.get("fund_shares_outstanding")
        if equity is not None and shares is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                bvps = np.where(
                    (shares != 0) & shares.notna(), equity / shares, np.nan
                )
                df["fund_pb_ratio"] = np.where(
                    (bvps != 0) & pd.notna(bvps), close / bvps, np.nan
                )
        else:
            df["fund_pb_ratio"] = np.nan

        # P/S ratio: (Close * shares_outstanding) / revenue_ttm
        rev_ttm = df.get("fund_revenue_ttm")
        if rev_ttm is not None and shares is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                mkt_cap = close * shares
                df["fund_ps_ratio"] = np.where(
                    (rev_ttm != 0) & rev_ttm.notna(),
                    mkt_cap / rev_ttm,
                    np.nan,
                )
        else:
            df["fund_ps_ratio"] = np.nan

        # FCF yield: free_cash_flow_ttm / (Close * shares_outstanding)
        fcf_ttm = df.get("fund_free_cash_flow_ttm")
        if fcf_ttm is not None and shares is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                mkt_cap = close * shares
                df["fund_fcf_yield"] = np.where(
                    (mkt_cap != 0) & mkt_cap.notna(),
                    fcf_ttm / mkt_cap,
                    np.nan,
                )
        else:
            df["fund_fcf_yield"] = np.nan

        return df

    def _compute_fundamental_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute non-price fundamental ratios."""

        # Current ratio
        ca = df.get("fund_current_assets")
        cl = df.get("fund_current_liabilities")
        if ca is not None and cl is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                df["fund_current_ratio"] = np.where(
                    (cl != 0) & cl.notna(), ca / cl, np.nan
                )
        else:
            df["fund_current_ratio"] = np.nan

        # Debt to equity
        debt = df.get("fund_total_debt")
        equity = df.get("fund_equity")
        if debt is not None and equity is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                df["fund_debt_to_equity"] = np.where(
                    (equity != 0) & equity.notna(), debt / equity, np.nan
                )
        else:
            df["fund_debt_to_equity"] = np.nan

        # Revenue growth YoY: (rev_q - rev_q-4) / abs(rev_q-4)
        rev = df.get("fund_revenue")
        if rev is not None:
            rev_lag4 = rev.shift(252)  # approx 4 quarters in trading days
            with np.errstate(divide="ignore", invalid="ignore"):
                df["fund_revenue_growth_yoy"] = np.where(
                    (rev_lag4 != 0) & rev_lag4.notna(),
                    (rev - rev_lag4) / np.abs(rev_lag4),
                    np.nan,
                )
        else:
            df["fund_revenue_growth_yoy"] = np.nan

        # Earnings growth YoY
        ni = df.get("fund_net_income")
        if ni is not None:
            ni_lag4 = ni.shift(252)
            with np.errstate(divide="ignore", invalid="ignore"):
                df["fund_earnings_growth_yoy"] = np.where(
                    (ni_lag4 != 0) & ni_lag4.notna(),
                    (ni - ni_lag4) / np.abs(ni_lag4),
                    np.nan,
                )
        else:
            df["fund_earnings_growth_yoy"] = np.nan

        # Margins
        if rev is not None:
            gp = df.get("fund_gross_profit")
            oi = df.get("fund_operating_income")

            with np.errstate(divide="ignore", invalid="ignore"):
                if gp is not None:
                    df["fund_gross_margin"] = np.where(
                        (rev != 0) & rev.notna(), gp / rev, np.nan
                    )
                else:
                    df["fund_gross_margin"] = np.nan

                if oi is not None:
                    df["fund_operating_margin"] = np.where(
                        (rev != 0) & rev.notna(), oi / rev, np.nan
                    )
                else:
                    df["fund_operating_margin"] = np.nan

                if ni is not None:
                    df["fund_net_margin"] = np.where(
                        (rev != 0) & rev.notna(), ni / rev, np.nan
                    )
                else:
                    df["fund_net_margin"] = np.nan
        else:
            df["fund_gross_margin"] = np.nan
            df["fund_operating_margin"] = np.nan
            df["fund_net_margin"] = np.nan

        # ROA: net_income_ttm / total_assets
        ni_ttm = df.get("fund_net_income_ttm")
        ta = df.get("fund_total_assets")
        if ni_ttm is not None and ta is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                df["fund_roa"] = np.where(
                    (ta != 0) & ta.notna(), ni_ttm / ta, np.nan
                )
        else:
            df["fund_roa"] = np.nan

        # ROE: net_income_ttm / equity
        if ni_ttm is not None and equity is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                df["fund_roe"] = np.where(
                    (equity != 0) & equity.notna(), ni_ttm / equity, np.nan
                )
        else:
            df["fund_roe"] = np.nan

        return df

    # ------------------------------------------------------------------
    # Derived metric computations
    # ------------------------------------------------------------------

    def _compute_piotroski_f(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Piotroski F-Score (0-9) from fundamental data.

        Nine binary criteria are summed. NaN inputs produce NaN for that
        criterion but the remaining criteria still contribute to the total.
        If *all* criteria are NaN the score is NaN.
        """
        ni_ttm = df.get("fund_net_income_ttm")
        ta = df.get("fund_total_assets")
        ocf_ttm = df.get("fund_operating_cf_ttm")
        debt = df.get("fund_total_debt")
        ca = df.get("fund_current_assets")
        cl = df.get("fund_current_liabilities")
        shares = df.get("fund_shares_outstanding")
        gp = df.get("fund_gross_profit")
        rev = df.get("fund_revenue")

        criteria: list[pd.Series] = []

        with np.errstate(divide="ignore", invalid="ignore"):
            # 1. ROA > 0
            if ni_ttm is not None and ta is not None:
                roa = pd.Series(
                    np.where((ta != 0) & ta.notna(), ni_ttm / ta, np.nan),
                    index=df.index,
                )
            else:
                roa = pd.Series(np.nan, index=df.index)
            criteria.append(pd.Series(
                np.where(roa.notna(), (roa > 0).astype(float), np.nan),
                index=df.index,
            ))

            # 2. CFO > 0
            if ocf_ttm is not None:
                criteria.append(pd.Series(
                    np.where(ocf_ttm.notna(), (ocf_ttm > 0).astype(float), np.nan),
                    index=df.index,
                ))
            else:
                criteria.append(pd.Series(np.nan, index=df.index))

            # 3. ROA change (current ROA > prior year ROA)
            roa_prev = roa.shift(252)
            criteria.append(pd.Series(
                np.where(roa.notna() & roa_prev.notna(), (roa > roa_prev).astype(float), np.nan),
                index=df.index,
            ))

            # 4. Accruals: CFO/TA > ROA
            if ocf_ttm is not None and ta is not None:
                cfo_ta = pd.Series(
                    np.where((ta != 0) & ta.notna(), ocf_ttm / ta, np.nan),
                    index=df.index,
                )
            else:
                cfo_ta = pd.Series(np.nan, index=df.index)
            criteria.append(pd.Series(
                np.where(cfo_ta.notna() & roa.notna(), (cfo_ta > roa).astype(float), np.nan),
                index=df.index,
            ))

            # 5. Leverage decreased: debt/assets < prior year
            if debt is not None and ta is not None:
                leverage = pd.Series(
                    np.where((ta != 0) & ta.notna(), debt / ta, np.nan),
                    index=df.index,
                )
            else:
                leverage = pd.Series(np.nan, index=df.index)
            lev_prev = leverage.shift(252)
            criteria.append(pd.Series(
                np.where(
                    leverage.notna() & lev_prev.notna(),
                    (leverage < lev_prev).astype(float),
                    np.nan,
                ),
                index=df.index,
            ))

            # 6. Liquidity improved: current ratio > prior year
            if ca is not None and cl is not None:
                cur_ratio = pd.Series(
                    np.where((cl != 0) & cl.notna(), ca / cl, np.nan),
                    index=df.index,
                )
            else:
                cur_ratio = pd.Series(np.nan, index=df.index)
            cr_prev = cur_ratio.shift(252)
            criteria.append(pd.Series(
                np.where(
                    cur_ratio.notna() & cr_prev.notna(),
                    (cur_ratio > cr_prev).astype(float),
                    np.nan,
                ),
                index=df.index,
            ))

            # 7. Dilution: shares_outstanding <= prior year
            if shares is not None:
                shares_prev = shares.shift(252)
                criteria.append(pd.Series(
                    np.where(
                        shares.notna() & shares_prev.notna(),
                        (shares <= shares_prev).astype(float),
                        np.nan,
                    ),
                    index=df.index,
                ))
            else:
                criteria.append(pd.Series(np.nan, index=df.index))

            # 8. Gross margin improved
            if gp is not None and rev is not None:
                gm = pd.Series(
                    np.where((rev != 0) & rev.notna(), gp / rev, np.nan),
                    index=df.index,
                )
            else:
                gm = pd.Series(np.nan, index=df.index)
            gm_prev = gm.shift(252)
            criteria.append(pd.Series(
                np.where(
                    gm.notna() & gm_prev.notna(),
                    (gm > gm_prev).astype(float),
                    np.nan,
                ),
                index=df.index,
            ))

            # 9. Asset turnover improved: revenue/total_assets > prior year
            if rev is not None and ta is not None:
                at = pd.Series(
                    np.where((ta != 0) & ta.notna(), rev / ta, np.nan),
                    index=df.index,
                )
            else:
                at = pd.Series(np.nan, index=df.index)
            at_prev = at.shift(252)
            criteria.append(pd.Series(
                np.where(
                    at.notna() & at_prev.notna(),
                    (at > at_prev).astype(float),
                    np.nan,
                ),
                index=df.index,
            ))

        # Stack and nansum; all-NaN → NaN
        stacked = np.column_stack(criteria)
        all_nan = np.all(np.isnan(stacked), axis=1)
        score = np.nansum(stacked, axis=1).astype(float)
        score[all_nan] = np.nan
        df["fund_piotroski_f"] = score

        return df

    def _compute_altman_z(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Altman Z-Score and zone classification.

        Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(OI/TA)
            + 0.6*(MktCap/TL) + 1.0*(Rev/TA)
        """
        close = df.get("Close")
        ta = df.get("fund_total_assets")
        re = df.get("fund_retained_earnings")
        oi = df.get("fund_operating_income")
        tl = df.get("fund_total_liabilities")
        rev = df.get("fund_revenue")
        ca = df.get("fund_current_assets")
        cl = df.get("fund_current_liabilities")
        shares = df.get("fund_shares_outstanding")

        df["fund_altman_z"] = np.nan
        df["fund_altman_zone"] = np.nan

        # Need at least total_assets and Close to compute anything useful
        if ta is None or close is None:
            return df

        with np.errstate(divide="ignore", invalid="ignore"):
            # Working capital / total assets
            if ca is not None and cl is not None:
                wc = ca - cl
                x1 = np.where((ta != 0) & ta.notna(), 1.2 * wc / ta, np.nan)
            else:
                x1 = np.full(len(df), np.nan)

            # Retained earnings / total assets
            if re is not None:
                x2 = np.where((ta != 0) & ta.notna(), 1.4 * re / ta, np.nan)
            else:
                x2 = np.full(len(df), np.nan)

            # Operating income / total assets
            if oi is not None:
                x3 = np.where((ta != 0) & ta.notna(), 3.3 * oi / ta, np.nan)
            else:
                x3 = np.full(len(df), np.nan)

            # Market cap / total liabilities
            if shares is not None and tl is not None:
                mkt_cap = close * shares
                x4 = np.where((tl != 0) & tl.notna(), 0.6 * mkt_cap / tl, np.nan)
            else:
                x4 = np.full(len(df), np.nan)

            # Revenue / total assets
            if rev is not None:
                x5 = np.where((ta != 0) & ta.notna(), 1.0 * rev / ta, np.nan)
            else:
                x5 = np.full(len(df), np.nan)

        components = np.column_stack([x1, x2, x3, x4, x5])
        all_nan = np.all(np.isnan(components), axis=1)
        z = np.nansum(components, axis=1).astype(float)
        z[all_nan] = np.nan
        df["fund_altman_z"] = z

        # Zone classification
        zone = pd.Series(np.nan, index=df.index, dtype=object)
        zone[z > 2.99] = "safe"
        zone[(z >= 1.8) & (z <= 2.99)] = "grey"
        zone[z < 1.8] = "distress"
        zone[np.isnan(z)] = np.nan
        df["fund_altman_zone"] = zone

        return df

    def _compute_shareholder_yield(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute buyback yield and shareholder yield.

        buyback_yield = (repurchased - issued) / market_cap
        shareholder_yield = (net_buyback + dividends_paid - stock_comp) / market_cap
        """
        close = df.get("Close")
        shares = df.get("fund_shares_outstanding")
        repurchased = df.get("fund_stock_repurchased_ttm")
        issued = df.get("fund_stock_issued_proceeds_ttm")
        div_paid = df.get("fund_dividends_paid_ttm")
        stock_comp = df.get("fund_stock_comp_ttm")

        df["fund_buyback_yield"] = np.nan
        df["fund_shareholder_yield"] = np.nan

        if close is None or shares is None:
            return df

        with np.errstate(divide="ignore", invalid="ignore"):
            mkt_cap = close * shares

            # Net buyback (fillna(0) for missing components)
            rep = repurchased.fillna(0) if repurchased is not None else pd.Series(0, index=df.index)
            iss = issued.fillna(0) if issued is not None else pd.Series(0, index=df.index)
            net_buyback = rep - iss

            df["fund_buyback_yield"] = np.where(
                (mkt_cap != 0) & mkt_cap.notna(),
                net_buyback / mkt_cap,
                np.nan,
            )

            # Shareholder yield
            div = div_paid.fillna(0) if div_paid is not None else pd.Series(0, index=df.index)
            sc = stock_comp.fillna(0) if stock_comp is not None else pd.Series(0, index=df.index)
            total = net_buyback + div - sc

            df["fund_shareholder_yield"] = np.where(
                (mkt_cap != 0) & mkt_cap.notna(),
                total / mkt_cap,
                np.nan,
            )

        return df

    def _compute_dividend_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute YoY dividend growth and payout ratio.

        div_growth_yoy = (DPS - DPS_prev) / abs(DPS_prev)
        payout_ratio   = dividends_paid_ttm / net_income_ttm
        """
        dps = df.get("fund_dividends_per_share")
        div_paid = df.get("fund_dividends_paid_ttm")
        ni_ttm = df.get("fund_net_income_ttm")

        with np.errstate(divide="ignore", invalid="ignore"):
            # Dividend growth YoY
            if dps is not None:
                dps_prev = dps.shift(252)
                df["fund_div_growth_yoy"] = np.where(
                    (dps_prev != 0) & dps_prev.notna(),
                    (dps - dps_prev) / np.abs(dps_prev),
                    np.nan,
                )
            else:
                df["fund_div_growth_yoy"] = np.nan

            # Payout ratio
            if div_paid is not None and ni_ttm is not None:
                df["fund_payout_ratio"] = np.where(
                    (ni_ttm != 0) & ni_ttm.notna(),
                    div_paid / ni_ttm,
                    np.nan,
                )
            else:
                df["fund_payout_ratio"] = np.nan

        return df

    # ------------------------------------------------------------------
    # Insider merge
    # ------------------------------------------------------------------

    def _merge_insider(
        self, symbol: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge insider trading data onto daily DataFrame."""
        insider = self.load_insider(symbol)
        if insider.empty:
            return df

        insider = insider.copy()
        insider["filed_date"] = pd.to_datetime(
            insider["filed_date"], errors="coerce"
        )
        insider.dropna(subset=["filed_date"], inplace=True)

        if insider.empty:
            return df

        # Compute rolling window aggregations for each trading day
        idx = df.index
        results: dict[str, list[float]] = {
            "insider_net_shares_30d": [],
            "insider_buy_count_90d": [],
            "insider_sell_count_90d": [],
            "insider_buy_ratio_90d": [],
            "insider_net_value_30d": [],
            "insider_officer_buys_90d": [],
        }

        officer_titles = {"ceo", "cfo", "coo", "chief executive", "chief financial", "chief operating"}

        for day in idx:
            day_ts = pd.Timestamp(day)
            d30 = day_ts - pd.Timedelta(days=30)
            d90 = day_ts - pd.Timedelta(days=90)

            mask_30 = (insider["filed_date"] >= d30) & (
                insider["filed_date"] <= day_ts
            )
            mask_90 = (insider["filed_date"] >= d90) & (
                insider["filed_date"] <= day_ts
            )

            sub_30 = insider[mask_30]
            sub_90 = insider[mask_90]

            # Net shares in 30 days
            net_shares = sub_30["shares"].sum() if not sub_30.empty else 0.0
            results["insider_net_shares_30d"].append(net_shares)

            # Buy/sell counts in 90 days
            if not sub_90.empty:
                buys_90 = (sub_90["shares"] > 0).sum()
                sells_90 = (sub_90["shares"] < 0).sum()
            else:
                buys_90 = 0
                sells_90 = 0
            results["insider_buy_count_90d"].append(float(buys_90))
            results["insider_sell_count_90d"].append(float(sells_90))

            # Buy ratio
            total_txn = buys_90 + sells_90
            ratio = buys_90 / total_txn if total_txn > 0 else np.nan
            results["insider_buy_ratio_90d"].append(ratio)

            # Net value in 30 days
            if not sub_30.empty and "price" in sub_30.columns:
                net_val = (sub_30["shares"] * sub_30["price"]).sum()
            else:
                net_val = 0.0
            results["insider_net_value_30d"].append(net_val)

            # Officer buys in 90 days
            if not sub_90.empty and "insider_title" in sub_90.columns:
                officer_mask = sub_90["insider_title"].str.lower().apply(
                    lambda t: any(
                        o in str(t) for o in officer_titles
                    )
                    if pd.notna(t)
                    else False
                )
                officer_buys = (
                    (sub_90[officer_mask]["shares"] > 0).sum()
                    if officer_mask.any()
                    else 0
                )
            else:
                officer_buys = 0
            results["insider_officer_buys_90d"].append(float(officer_buys))

        for col, vals in results.items():
            df[col] = vals

        return df

    # ------------------------------------------------------------------
    # Institutional merge
    # ------------------------------------------------------------------

    def _merge_institutional(
        self, symbol: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge institutional holdings data onto daily DataFrame."""
        inst = self.load_institutional(symbol)
        if inst.empty:
            return df

        inst = inst.copy()
        inst["filed_date"] = pd.to_datetime(inst["filed_date"], errors="coerce")
        inst.dropna(subset=["filed_date"], inplace=True)

        if inst.empty:
            return df

        inst.sort_values("filed_date", inplace=True)

        # QoQ changes
        inst["_holders_prev"] = inst["total_holders"].shift(1)
        inst["_shares_prev"] = inst["total_shares"].shift(1)

        inst["inst_holders_change_qoq"] = (
            inst["total_holders"] - inst["_holders_prev"]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            inst["inst_shares_change_pct"] = np.where(
                (inst["_shares_prev"] != 0) & inst["_shares_prev"].notna(),
                (inst["total_shares"] - inst["_shares_prev"]) / inst["_shares_prev"],
                np.nan,
            )

        # Ownership concentration (placeholder — actual top-5 requires
        # per-holder data which 13F aggregation doesn't provide)
        inst["inst_ownership_concentration"] = np.nan

        # Merge onto daily via merge_asof
        merge_cols = [
            "filed_date",
            "inst_holders_change_qoq",
            "inst_shares_change_pct",
            "inst_ownership_concentration",
        ]
        inst_merge = inst[merge_cols].copy()
        inst_merge.sort_values("filed_date", inplace=True)
        inst_merge.reset_index(drop=True, inplace=True)

        df_reset = df.reset_index()
        date_col = df_reset.columns[0]
        df_reset[date_col] = pd.to_datetime(df_reset[date_col])
        df_reset.sort_values(date_col, inplace=True)

        merged = pd.merge_asof(
            df_reset,
            inst_merge,
            left_on=date_col,
            right_on="filed_date",
            direction="backward",
        )
        merged.set_index(date_col, inplace=True)
        merged.index.name = df.index.name

        if "filed_date" in merged.columns:
            merged.drop(columns=["filed_date"], inplace=True)

        return merged

    # ------------------------------------------------------------------
    # Events merge
    # ------------------------------------------------------------------

    def _merge_events(
        self, symbol: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge 8-K event flags onto daily DataFrame."""
        events = self.load_events(symbol)
        if events.empty:
            return df

        events = events.copy()
        events["filed_date"] = pd.to_datetime(
            events["filed_date"], errors="coerce"
        )
        events.dropna(subset=["filed_date"], inplace=True)

        if events.empty:
            return df

        idx = df.index
        results: dict[str, list[float]] = {
            "event_earnings_recent": [],
            "event_acquisition_recent": [],
            "event_officer_change_recent": [],
            "event_restatement_ever": [],
            "event_count_90d": [],
        }

        for day in idx:
            day_ts = pd.Timestamp(day)

            # Earnings: last 5 trading days (~7 calendar days)
            d5 = day_ts - pd.Timedelta(days=7)
            mask_5 = (events["filed_date"] >= d5) & (
                events["filed_date"] <= day_ts
            )
            has_earn = (
                events[mask_5]["has_earnings"].any() if mask_5.any() else False
            )
            results["event_earnings_recent"].append(1.0 if has_earn else 0.0)

            # Acquisition: last 10 trading days (~14 calendar days)
            d10 = day_ts - pd.Timedelta(days=14)
            mask_10 = (events["filed_date"] >= d10) & (
                events["filed_date"] <= day_ts
            )
            has_acq = (
                events[mask_10]["has_acquisition"].any()
                if mask_10.any()
                else False
            )
            results["event_acquisition_recent"].append(1.0 if has_acq else 0.0)

            # Officer change: last 20 trading days (~28 calendar days)
            d20 = day_ts - pd.Timedelta(days=28)
            mask_20 = (events["filed_date"] >= d20) & (
                events["filed_date"] <= day_ts
            )
            has_off = (
                events[mask_20]["has_officer_change"].any()
                if mask_20.any()
                else False
            )
            results["event_officer_change_recent"].append(
                1.0 if has_off else 0.0
            )

            # Restatement: trailing 252 trading days (~365 calendar days)
            d252 = day_ts - pd.Timedelta(days=365)
            mask_252 = (events["filed_date"] >= d252) & (
                events["filed_date"] <= day_ts
            )
            has_rest = (
                events[mask_252]["has_restatement"].any()
                if mask_252.any()
                else False
            )
            results["event_restatement_ever"].append(
                1.0 if has_rest else 0.0
            )

            # Event count in 90 days
            d90 = day_ts - pd.Timedelta(days=90)
            mask_90 = (events["filed_date"] >= d90) & (
                events["filed_date"] <= day_ts
            )
            results["event_count_90d"].append(float(mask_90.sum()))

        for col, vals in results.items():
            df[col] = vals

        return df

    # ------------------------------------------------------------------
    # Backward-compatible get() method
    # ------------------------------------------------------------------

    def get(self, symbol: str, field: str, as_of: date) -> float | None:
        """Return the latest fundamental value on or before *as_of*.

        Backward-compatible with the original FundamentalDataManager.
        Checks the CSV source first, then falls back to EDGAR financials.
        """
        # CSV source takes priority
        if self._csv is not None:
            val = self._csv.get(symbol, field, as_of)
            if val is not None:
                return val

        # Fallback to EDGAR financials
        if self._financials_source is not None or self._financials_cache is not None:
            fin = self.load_financials(symbol)
            if not fin.empty and "metric" in fin.columns:
                matched = fin[fin["metric"] == field].copy()
                if not matched.empty:
                    matched["filed_date"] = pd.to_datetime(
                        matched["filed_date"], errors="coerce"
                    )
                    matched.dropna(subset=["filed_date"], inplace=True)
                    as_of_ts = pd.Timestamp(as_of)
                    valid = matched[matched["filed_date"] <= as_of_ts]
                    if not valid.empty:
                        return float(
                            valid.sort_values("filed_date").iloc[-1]["value"]
                        )

        return None


# ---------------------------------------------------------------------------
# Backward compatibility alias
# ---------------------------------------------------------------------------

# FundamentalDataManager preserved as the original CSV-only class.
# Code that imports FundamentalDataManager will continue to work unchanged.
FundamentalDataManager = _CSVFundamentalData
