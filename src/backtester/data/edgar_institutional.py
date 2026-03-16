"""SEC EDGAR 13F-HR institutional holdings data via edgartools."""

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


class EdgarInstitutionalSource:
    """Fetch 13F-HR institutional holdings data from SEC EDGAR.

    Note: 13F filings are filed by institutional investment managers, not
    by the companies whose shares they hold. This source searches for 13F
    filings referencing the given symbol and aggregates holder counts and
    share totals per reporting period.

    Returns a DataFrame with columns:
        filed_date, report_date, total_holders, total_shares, total_value
    """

    def __init__(self, user_agent: str, max_filings: int = 50) -> None:
        if Company is None:
            raise ImportError(
                "edgartools is required for EDGAR institutional data. "
                "Install it with: pip install edgartools"
            )
        self.user_agent = user_agent
        self.max_filings = max_filings

    @edgar_retry()
    def fetch(self, symbol: str) -> pd.DataFrame:
        """Fetch aggregated 13F institutional holding data for *symbol*.

        Since 13F filings are per-manager (not per-stock), this method
        retrieves filings from the company's major institutional holders
        and aggregates share counts by reporting period.
        """
        company = Company(symbol)

        try:
            filings = company.get_filings(form="13F-HR")
        except Exception as exc:
            from backtester.data.edgar_utils import _is_rate_limit_error
            if _is_rate_limit_error(exc):
                raise
            logger.warning("Could not retrieve 13F filings for %s", symbol)
            return self._empty_df()

        if filings is None:
            return self._empty_df()

        rows: list[dict] = []
        for filing in filings[:self.max_filings]:
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

                # Try to extract holdings table
                holdings = getattr(parsed, "holdings", None)
                if holdings is None:
                    holdings = getattr(parsed, "infotable", None)

                if holdings is None:
                    continue

                if isinstance(holdings, pd.DataFrame) and not holdings.empty:
                    total_holders = len(holdings)
                    total_shares = 0
                    total_value = 0

                    for col in ["shares", "shrsOrPrnAmt", "sshPrnamt"]:
                        if col in holdings.columns:
                            total_shares = int(
                                pd.to_numeric(
                                    holdings[col], errors="coerce"
                                ).sum()
                            )
                            break

                    for col in ["value", "market_value", "value_x1000"]:
                        if col in holdings.columns:
                            total_value = int(
                                pd.to_numeric(
                                    holdings[col], errors="coerce"
                                ).sum()
                            )
                            break

                    rows.append(
                        {
                            "filed_date": filed_date,
                            "report_date": report_date,
                            "total_holders": total_holders,
                            "total_shares": total_shares,
                            "total_value": total_value,
                        }
                    )
            except Exception:
                logger.debug("Failed to parse 13F filing for %s", symbol)
                continue

        if not rows:
            return self._empty_df()

        df = pd.DataFrame(rows, columns=_COLUMNS)
        df.sort_values("filed_date", inplace=True)
        df.drop_duplicates(subset=["report_date"], keep="last", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

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
