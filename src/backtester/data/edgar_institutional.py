"""SEC EDGAR 13F-HR institutional holdings data via edgartools.

13F-HR filings are filed BY institutional investment managers (e.g.,
Berkshire Hathaway, BlackRock), not by the companies whose shares they
hold.  This module queries a configurable list of top managers' 13F
filings and aggregates holdings for a target symbol.
"""

import logging
from datetime import date

import pandas as pd

from backtester.data.edgar_utils import edgar_retry

logger = logging.getLogger(__name__)

try:
    from edgar import Company
except ImportError:
    Company = None  # type: ignore[assignment,misc]

_COLUMNS = [
    "filed_date",
    "report_date",
    "total_holders",
    "total_shares",
    "total_value",
]

# Top institutional managers by CIK.
# CIK values are best-effort; the architecture matters more than
# exact CIK accuracy — they are used at runtime for SEC EDGAR lookups.
_DEFAULT_MANAGERS: dict[str, str] = {
    "1067983": "Berkshire Hathaway",
    "1364742": "BlackRock",
    "102909": "Vanguard Group",
    "1350694": "Citadel Advisors",
    "1037389": "Renaissance Technologies",
    "1336528": "Bridgewater Associates",
    "1061768": "JPMorgan Chase",
    "921669": "Fidelity Management",
    "1166559": "DE Shaw",
    "1423053": "Tiger Global Management",
    "1649339": "Point72 Asset Management",
    "1159159": "Baupost Group",
    "1061165": "Two Sigma Investments",
    "40729": "Goldman Sachs",
    "1133137": "Millennium Management",
    "1510446": "Coatue Management",
    "1541617": "Pershing Square Capital",
    "1039565": "Third Point",
    "1099281": "Viking Global Investors",
    "1484532": "Lone Pine Capital",
}


class EdgarInstitutionalSource:
    """Fetch 13F-HR institutional holdings data from SEC EDGAR.

    Queries top institutional managers' 13F filings and aggregates
    holdings data for the target symbol across managers.

    Returns a DataFrame with columns:
        filed_date, report_date, total_holders, total_shares, total_value
    """

    def __init__(
        self,
        user_agent: str,
        max_filings: int = 50,
        managers: dict[str, str] | None = None,
        filings_per_manager: int = 4,
    ) -> None:
        if Company is None:
            raise ImportError(
                "edgartools is required for EDGAR institutional data. "
                "Install it with: pip install edgartools"
            )
        self.user_agent = user_agent
        self.max_filings = max_filings
        self._managers = managers or _DEFAULT_MANAGERS
        self._filings_per_manager = filings_per_manager

    @edgar_retry()
    def fetch(self, symbol: str) -> pd.DataFrame:
        """Fetch aggregated 13F institutional holding data for *symbol*.

        Queries top institutional managers' 13F filings and aggregates
        holdings data for the target symbol across managers.
        """
        rows: list[dict] = []

        for manager_cik, manager_name in self._managers.items():
            try:
                manager = Company(manager_cik)
                filings = manager.get_filings(form="13F-HR")
                if filings is None:
                    continue

                for filing in filings[: self._filings_per_manager]:
                    try:
                        filed_date = self._parse_date(
                            getattr(filing, "filing_date", None)
                            or getattr(filing, "filed", None)
                        )
                        report_date = self._parse_date(
                            getattr(filing, "report_date", None)
                            or getattr(filing, "period_of_report", None)
                        ) or filed_date

                        parsed = filing.obj()
                        if parsed is None:
                            continue

                        # Extract holdings table from the parsed 13F
                        holdings_df = self._extract_holdings(parsed)
                        if holdings_df is None or holdings_df.empty:
                            continue

                        # Filter for the target symbol
                        symbol_holdings = self._filter_for_symbol(
                            holdings_df, symbol
                        )
                        if symbol_holdings.empty:
                            continue

                        # Aggregate this manager's position in the symbol
                        total_shares = self._sum_column(
                            symbol_holdings,
                            [
                                "SharesPrnAmount",
                                "shares",
                                "shrsOrPrnAmt",
                                "sshPrnamt",
                                "SHARES",
                                "Shares",
                            ],
                        )
                        total_value = self._sum_column(
                            symbol_holdings,
                            [
                                "Value",
                                "value",
                                "market_value",
                                "value_x1000",
                                "VALUE",
                            ],
                        )

                        rows.append(
                            {
                                "filed_date": filed_date,
                                "report_date": report_date,
                                "manager_cik": manager_cik,
                                "manager_name": manager_name,
                                "shares_held": total_shares,
                                "value_held": total_value,
                            }
                        )
                    except Exception:
                        logger.debug(
                            "Failed to parse 13F from %s", manager_name
                        )
                        continue
            except Exception as exc:
                from backtester.data.edgar_utils import _is_rate_limit_error

                if _is_rate_limit_error(exc):
                    raise
                logger.debug(
                    "Failed to fetch 13F filings for %s", manager_name
                )
                continue

        if not rows:
            return self._empty_df()

        # Aggregate per-manager rows into per-period summary
        df = pd.DataFrame(rows)
        aggregated = self._aggregate_by_period(df)
        return aggregated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_holdings(parsed: object) -> pd.DataFrame | None:
        """Try multiple attributes to get a holdings DataFrame from a 13F."""
        for attr in ("infotable", "holdings", "data"):
            obj = getattr(parsed, attr, None)
            if obj is not None:
                if isinstance(obj, pd.DataFrame) and not obj.empty:
                    return obj
                # Some edgartools versions use a wrapper with to_dataframe()
                if hasattr(obj, "to_dataframe"):
                    try:
                        result = obj.to_dataframe()
                        if isinstance(result, pd.DataFrame) and not result.empty:
                            return result
                    except Exception:
                        continue

        # Last resort: the parsed object itself may be convertible
        if hasattr(parsed, "to_dataframe"):
            try:
                result = parsed.to_dataframe()
                if isinstance(result, pd.DataFrame) and not result.empty:
                    return result
            except Exception:
                pass

        return None

    @staticmethod
    def _filter_for_symbol(
        holdings_df: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Filter a holdings DataFrame for rows matching *symbol*.

        Tries matching on common column names used in 13F data:
        ticker/symbol columns (exact match) and issuer name columns
        (case-insensitive partial match).
        """
        symbol_upper = symbol.upper()
        symbol_lower = symbol.lower()

        # Try exact ticker/symbol match first
        for col in ("ticker", "symbol", "TICKER", "SYMBOL", "Ticker", "Symbol"):
            if col in holdings_df.columns:
                mask = holdings_df[col].astype(str).str.upper() == symbol_upper
                if mask.any():
                    return holdings_df[mask]

        # Try CUSIP match (not implemented — would need a CUSIP lookup)

        # Try issuer name match (partial, case-insensitive)
        for col in (
            "nameOfIssuer",
            "issuer",
            "name",
            "ISSUER",
            "Name",
            "issuerName",
        ):
            if col in holdings_df.columns:
                col_lower = holdings_df[col].astype(str).str.lower()
                mask = col_lower.str.contains(symbol_lower, na=False)
                if mask.any():
                    return holdings_df[mask]

        return pd.DataFrame()

    @staticmethod
    def _sum_column(df: pd.DataFrame, candidate_cols: list[str]) -> int:
        """Sum the first matching column from *candidate_cols*."""
        for col in candidate_cols:
            if col in df.columns:
                return int(pd.to_numeric(df[col], errors="coerce").sum())
        return 0

    @staticmethod
    def _aggregate_by_period(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-manager rows into per-report_date summary.

        Output columns: filed_date, report_date, total_holders,
        total_shares, total_value.

        - total_holders: number of unique managers holding the stock
        - total_shares: sum of shares across managers
        - total_value: sum of value across managers
        - filed_date: latest filing date for that report period
        """
        grouped = df.groupby("report_date").agg(
            filed_date=("filed_date", "max"),
            total_holders=("manager_cik", "nunique"),
            total_shares=("shares_held", "sum"),
            total_value=("value_held", "sum"),
        )
        grouped.reset_index(inplace=True)

        # Ensure correct column order
        result = grouped[_COLUMNS].copy()
        result.sort_values("filed_date", inplace=True)
        result.drop_duplicates(subset=["report_date"], keep="last", inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    @staticmethod
    def _parse_date(val: object) -> date | None:
        if val is None:
            return None
        if isinstance(val, date):
            return val
        try:
            return date.fromisoformat(str(val))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        return pd.DataFrame(columns=_COLUMNS)
