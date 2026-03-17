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

# Bug 3 fix: Only open-market purchases (P) and sales (S).
# Codes A (award/grant) and M (derivative exercise) are RSU vestings /
# stock awards that are not genuine open-market trades.
_TRANSACTION_CODES = {"P", "S"}

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

# Column name candidates for DataFrame-based Form 4 parsing.
# edgartools versions expose different column names; we try each in order.
_COL_CANDIDATES = {
    "transaction_code": ["transaction_code", "code", "Code", "TransactionCode"],
    "shares": [
        "transaction_shares", "shares", "Shares", "TransactionShares", "Amount",
    ],
    "price": [
        "transaction_price_per_share", "price", "Price", "PricePerShare",
    ],
    "transaction_date": ["transaction_date", "Date", "TransactionDate"],
    "acquired_disposed": [
        "acquired_disposed_code", "AcquiredDisposedCode", "acquired_disposed",
    ],
    "shares_after": [
        "shares_owned_following", "Remaining Shares",
        "SharesOwnedFollowingTransaction",
        "sharesOwnedFollowingTransaction",
    ],
    "direct_indirect": [
        "direct_or_indirect_ownership", "DirectOrIndirectOwnership",
        "ownership_nature",
    ],
    "insider_name": ["Insider", "insider_name", "owner_name"],
    "insider_title": ["Position", "insider_title", "owner_title"],
}


def _resolve_col(df: pd.DataFrame, key: str) -> str | None:
    """Return the first matching column name from *df* for logical *key*."""
    for candidate in _COL_CANDIDATES.get(key, []):
        if candidate in df.columns:
            return candidate
    return None


class EdgarInsiderSource:
    """Fetch Form 4 insider trading data from SEC EDGAR.

    Returns a DataFrame with columns:
        filed_date, transaction_date, insider_name, insider_title,
        transaction_type (P/S), shares (positive=buy, negative=sell),
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

        # Resolve the target company's CIK for issuer matching (Bug 1).
        target_cik = getattr(company, "cik", None)

        rows: list[dict] = []
        for filing in filings[:self.max_filings]:
            try:
                parsed = filing.obj()
                if parsed is None:
                    continue

                # Bug 1 fix: verify the issuer matches the target company.
                # Form 4 filings returned by get_filings(form="4") may be
                # filings where the company is the *filer* (e.g. an
                # institutional investor filing about trades in other
                # companies' stock).  We only want filings where the
                # company is the *issuer* — i.e. insider trades of *this*
                # company's stock.
                issuer = getattr(parsed, "issuer", None)
                if issuer is not None and target_cik is not None:
                    issuer_cik = getattr(issuer, "cik", None)
                    if issuer_cik is not None and str(issuer_cik).lstrip("0") != str(target_cik).lstrip("0"):
                        logger.debug(
                            "Skipping Form 4: issuer CIK %s != target CIK %s",
                            issuer_cik, target_cik,
                        )
                        continue

                filed_date = self._parse_date(
                    getattr(filing, "filing_date", None)
                    or getattr(filing, "filed", None)
                )

                owner_name = getattr(parsed, "owner_name", "") or ""
                owner_title = getattr(parsed, "owner_title", "") or ""

                # Bug 2 fix: try multiple edgartools APIs to extract
                # transactions.  Newer versions return a DataFrame via
                # ``non_derivative_table`` or ``to_dataframe()``; older
                # versions expose iterable attributes.
                txn_df = getattr(parsed, "non_derivative_table", None)

                # If non_derivative_table is not a usable DataFrame,
                # try converting it or fall back to parsed.to_dataframe().
                if txn_df is not None and not isinstance(txn_df, pd.DataFrame):
                    if hasattr(txn_df, "to_dataframe"):
                        try:
                            txn_df = txn_df.to_dataframe()
                        except Exception:
                            txn_df = None
                    else:
                        txn_df = None

                if txn_df is None or (
                    isinstance(txn_df, pd.DataFrame) and txn_df.empty
                ):
                    try:
                        txn_df = parsed.to_dataframe()
                    except (AttributeError, Exception):
                        txn_df = None

                if isinstance(txn_df, pd.DataFrame) and not txn_df.empty:
                    # DataFrame-based parsing path
                    self._parse_transactions_df(
                        txn_df, filed_date, owner_name, owner_title, rows,
                    )
                else:
                    # Legacy iterable-based parsing path
                    transactions = (
                        getattr(parsed, "transactions", None)
                        or getattr(parsed, "non_derivative_transactions", None)
                        or []
                    )
                    self._parse_transactions_iter(
                        transactions, filed_date, owner_name, owner_title, rows,
                    )

            except Exception:
                logger.debug("Failed to parse Form 4 filing for %s", symbol)
                continue

        if not rows:
            return self._empty_df()

        df = pd.DataFrame(rows, columns=_COLUMNS)
        df.sort_values("filed_date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ------------------------------------------------------------------
    # Transaction parsing helpers
    # ------------------------------------------------------------------

    def _parse_transactions_df(
        self,
        txn_df: pd.DataFrame,
        filed_date: date | None,
        owner_name: str,
        owner_title: str,
        rows: list[dict],
    ) -> None:
        """Parse transactions from a DataFrame (newer edgartools API)."""
        code_col = _resolve_col(txn_df, "transaction_code")
        shares_col = _resolve_col(txn_df, "shares")
        price_col = _resolve_col(txn_df, "price")
        date_col = _resolve_col(txn_df, "transaction_date")
        ad_col = _resolve_col(txn_df, "acquired_disposed")
        sa_col = _resolve_col(txn_df, "shares_after")
        di_col = _resolve_col(txn_df, "direct_indirect")
        name_col = _resolve_col(txn_df, "insider_name")
        title_col = _resolve_col(txn_df, "insider_title")

        for _, row in txn_df.iterrows():
            try:
                code = str(row.get(code_col, "")) if code_col else ""
                if code not in _TRANSACTION_CODES:
                    continue

                shares_raw = float(row.get(shares_col, 0) or 0) if shares_col else 0.0
                price_raw = float(row.get(price_col, 0) or 0) if price_col else 0.0

                # Bug 3 fix: skip zero-price transactions (grants/awards)
                if price_raw == 0:
                    continue

                shares_after_raw = (
                    float(row.get(sa_col, 0) or 0) if sa_col else 0.0
                )

                txn_date = self._parse_date(
                    row.get(date_col) if date_col else None
                )

                is_direct = True
                if di_col:
                    is_direct = str(row.get(di_col, "D")) == "D"

                # Sign convention: buys positive, sells negative
                acquired_disposed = ""
                if ad_col:
                    acquired_disposed = str(row.get(ad_col, "") or "")
                if acquired_disposed == "D" or code == "S":
                    shares_raw = -abs(shares_raw)
                else:
                    shares_raw = abs(shares_raw)

                # Prefer name/title from DataFrame if available
                row_name = owner_name
                if name_col:
                    row_name = str(row.get(name_col, "") or "") or owner_name
                row_title = owner_title
                if title_col:
                    row_title = str(row.get(title_col, "") or "") or owner_title

                rows.append(
                    {
                        "filed_date": filed_date,
                        "transaction_date": txn_date or filed_date,
                        "insider_name": row_name,
                        "insider_title": row_title,
                        "transaction_type": code,
                        "shares": shares_raw,
                        "price": price_raw,
                        "shares_after": shares_after_raw,
                        "is_direct": is_direct,
                    }
                )
            except (ValueError, TypeError, AttributeError):
                continue

    def _parse_transactions_iter(
        self,
        transactions,
        filed_date: date | None,
        owner_name: str,
        owner_title: str,
        rows: list[dict],
    ) -> None:
        """Parse transactions from an iterable of objects (legacy API)."""
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

                # Bug 3 fix: skip zero-price transactions (grants/awards)
                if price_raw == 0:
                    continue

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
