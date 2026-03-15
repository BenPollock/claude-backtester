"""SEC EDGAR financial statement data via edgartools (10-K/10-Q XBRL)."""

import logging
import time
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from edgar import Company, set_identity
except ImportError:
    Company = None  # type: ignore[assignment,misc]
    set_identity = None  # type: ignore[assignment]

# Normalized metric name -> ordered list of XBRL tag fallbacks.
# First tag with data wins.
TAG_MAP: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "operating_income": ["OperatingIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "ebitda": ["EBITDA"],
    "total_assets": ["Assets"],
    "total_debt": ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "operating_cf": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ],
    "research_dev": ["ResearchAndDevelopmentExpense"],
}


class EdgarFundamentalSource:
    """Fetch 10-K/10-Q XBRL financial data from SEC EDGAR.

    Uses the ``edgartools`` library.  SEC requires a User-Agent header
    identifying the caller (name + email).

    The ``fetch`` method returns a DataFrame with columns:
        metric, period_end, filed_date, value, form
    """

    def __init__(self, user_agent: str) -> None:
        if Company is None:
            raise ImportError(
                "edgartools is required for EDGAR data. "
                "Install it with: pip install edgartools"
            )
        self.user_agent = user_agent
        set_identity(user_agent)

    def fetch(self, symbol: str) -> pd.DataFrame:
        """Fetch financial statement data for *symbol* from EDGAR.

        Returns a DataFrame with columns:
            metric (str), period_end (date), filed_date (date),
            value (float), form (str)
        """
        company = Company(symbol)
        time.sleep(0.1)  # SEC rate-limit courtesy

        try:
            facts = company.get_facts()
        except Exception:
            logger.warning("Could not retrieve XBRL facts for %s", symbol)
            return self._empty_df()

        if facts is None:
            return self._empty_df()

        rows: list[dict] = []
        for metric, tags in TAG_MAP.items():
            resolved = self._resolve_tag(facts, metric, tags)
            rows.extend(resolved)

        if not rows:
            return self._empty_df()

        df = pd.DataFrame(rows)
        # Keep only 10-K and 10-Q filings
        df = df[df["form"].isin(("10-Q", "10-K"))].copy()

        # Dedup by (metric, period_end), keeping the latest filing
        df.sort_values("filed_date", inplace=True)
        df.drop_duplicates(subset=["metric", "period_end"], keep="last", inplace=True)
        df.sort_values(["metric", "period_end"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_tag(
        facts: object, metric: str, tags: list[str]
    ) -> list[dict]:
        """Try XBRL tags in order; return records from first tag with data."""
        for tag in tags:
            try:
                # edgartools facts lookup — API may vary across versions
                concept = facts.get(tag)
                if concept is None or concept.empty:
                    continue

                records: list[dict] = []
                for _, row in concept.iterrows():
                    try:
                        period_end = (
                            row.get("end") or row.get("period_end") or row.get("period")
                        )
                        filed = row.get("filed") or row.get("filed_date")
                        val = row.get("val") or row.get("value")
                        form = row.get("form", "")

                        if period_end is None or val is None:
                            continue

                        if isinstance(period_end, str):
                            period_end = date.fromisoformat(period_end)
                        if isinstance(filed, str):
                            filed = date.fromisoformat(filed)

                        records.append(
                            {
                                "metric": metric,
                                "period_end": period_end,
                                "filed_date": filed,
                                "value": float(val),
                                "form": str(form),
                            }
                        )
                    except (ValueError, TypeError, AttributeError):
                        continue

                if records:
                    return records
            except (AttributeError, KeyError, TypeError):
                continue
        return []

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["metric", "period_end", "filed_date", "value", "form"]
        )
