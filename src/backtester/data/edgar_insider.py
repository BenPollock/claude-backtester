"""SEC EDGAR Form 4 insider trading data via edgartools."""

import logging
from datetime import date

import pandas as pd

from backtester.data.edgar_utils import edgar_retry

logger = logging.getLogger(__name__)

try:
    from edgar import Company
except ImportError:
    Company = None  # type: ignore[assignment,misc]

# Form 4 transaction type codes
_TRANSACTION_CODES = {"P", "S", "A", "M"}

_COLUMNS = [
    "filed_date",
    "transaction_date",
    "insider_name",
    "insider_title",
    "transaction_type",
    "shares",
    "price",
    "shares_after",
    "is_direct",
]


class EdgarInsiderSource:
    """Fetch Form 4 insider trading data from SEC EDGAR.

    Returns a DataFrame with columns:
        filed_date, transaction_date, insider_name, insider_title,
        transaction_type (P/S/A/M), shares (positive=buy, negative=sell),
        price, shares_after, is_direct
    """

    def __init__(self, user_agent: str, max_filings: int = 50) -> None:
        if Company is None:
            raise ImportError(
                "edgartools is required for EDGAR insider data. "
                "Install it with: pip install edgartools"
            )
        self.user_agent = user_agent
        self.max_filings = max_filings

    @edgar_retry()
    def fetch(self, symbol: str) -> pd.DataFrame:
        """Fetch Form 4 insider transactions for *symbol*."""
        company = Company(symbol)

        try:
            filings = company.get_filings(form="4")
        except Exception as exc:
            from backtester.data.edgar_utils import _is_rate_limit_error
            if _is_rate_limit_error(exc):
                raise
            logger.warning("Could not retrieve Form 4 filings for %s", symbol)
            return self._empty_df()

        if filings is None:
            return self._empty_df()

        rows: list[dict] = []
        for filing in filings[:self.max_filings]:
            try:
                parsed = filing.obj()
                if parsed is None:
                    continue

                filed_date = self._parse_date(
                    getattr(filing, "filing_date", None)
                    or getattr(filing, "filed", None)
                )

                # edgartools Form 4 parsing — extract transactions
                transactions = (
                    getattr(parsed, "transactions", None)
                    or getattr(parsed, "non_derivative_transactions", None)
                    or []
                )

                owner_name = getattr(parsed, "owner_name", "") or ""
                owner_title = getattr(parsed, "owner_title", "") or ""

                for txn in transactions:
                    try:
                        code = getattr(txn, "transaction_code", "") or ""
                        if code not in _TRANSACTION_CODES:
                            continue

                        shares_raw = float(
                            getattr(txn, "transaction_shares", 0) or 0
                        )
                        price_raw = float(
                            getattr(txn, "transaction_price_per_share", 0) or 0
                        )
                        shares_after_raw = float(
                            getattr(txn, "shares_owned_following", 0) or 0
                        )

                        txn_date = self._parse_date(
                            getattr(txn, "transaction_date", None)
                        )

                        is_direct = (
                            getattr(txn, "direct_or_indirect_ownership", "D")
                            == "D"
                        )

                        # Sign convention: buys positive, sells negative
                        acquired_disposed = getattr(
                            txn, "acquired_disposed_code", ""
                        ) or ""
                        if acquired_disposed == "D" or code == "S":
                            shares_raw = -abs(shares_raw)
                        else:
                            shares_raw = abs(shares_raw)

                        rows.append(
                            {
                                "filed_date": filed_date,
                                "transaction_date": txn_date or filed_date,
                                "insider_name": owner_name,
                                "insider_title": owner_title,
                                "transaction_type": code,
                                "shares": shares_raw,
                                "price": price_raw,
                                "shares_after": shares_after_raw,
                                "is_direct": is_direct,
                            }
                        )
                    except (ValueError, TypeError, AttributeError):
                        continue
            except Exception:
                logger.debug("Failed to parse Form 4 filing for %s", symbol)
                continue

        if not rows:
            return self._empty_df()

        df = pd.DataFrame(rows, columns=_COLUMNS)
        df.sort_values("filed_date", inplace=True)
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
