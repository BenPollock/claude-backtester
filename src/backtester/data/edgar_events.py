"""SEC EDGAR 8-K material event data via edgartools."""

import logging
import time
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from edgar import Company
except ImportError:
    Company = None  # type: ignore[assignment,misc]

# 8-K item number -> event category
_ITEM_MAP: dict[str, str] = {
    "2.02": "earnings",
    "2.01": "acquisition",
    "5.02": "officer_change",
    "4.02": "restatement",
    "1.01": "material_agreement",
    "3.01": "delisting",
    "1.02": "bankruptcy",
    "1.03": "bankruptcy",
}

_BOOLEAN_FLAGS = [
    "has_earnings",
    "has_acquisition",
    "has_officer_change",
    "has_restatement",
    "has_material_agreement",
    "has_delisting_notice",
    "has_bankruptcy",
]

_COLUMNS = [
    "filed_date",
    "event_date",
    "items",
    "item_descriptions",
    *_BOOLEAN_FLAGS,
]


class EdgarEventSource:
    """Fetch 8-K material event filings from SEC EDGAR.

    Returns a DataFrame with columns:
        filed_date, event_date, items (list), item_descriptions (list),
        has_earnings, has_acquisition, has_officer_change, has_restatement,
        has_material_agreement, has_delisting_notice, has_bankruptcy
    """

    def __init__(self, user_agent: str) -> None:
        if Company is None:
            raise ImportError(
                "edgartools is required for EDGAR event data. "
                "Install it with: pip install edgartools"
            )
        self.user_agent = user_agent

    def fetch(self, symbol: str) -> pd.DataFrame:
        """Fetch 8-K events for *symbol*."""
        company = Company(symbol)
        time.sleep(0.1)

        try:
            filings = company.get_filings(form="8-K")
        except Exception:
            logger.warning("Could not retrieve 8-K filings for %s", symbol)
            return self._empty_df()

        if filings is None:
            return self._empty_df()

        rows: list[dict] = []
        for filing in filings:
            time.sleep(0.1)
            try:
                filed_date = self._parse_date(
                    getattr(filing, "filing_date", None)
                    or getattr(filing, "filed", None)
                )

                # Event date may differ from filing date
                event_date = self._parse_date(
                    getattr(filing, "report_date", None)
                    or getattr(filing, "period_of_report", None)
                ) or filed_date

                # Extract 8-K items from filing
                items_list: list[str] = []
                descriptions: list[str] = []

                raw_items = getattr(filing, "items", None)
                if raw_items:
                    if isinstance(raw_items, str):
                        items_list = [
                            s.strip() for s in raw_items.split(",") if s.strip()
                        ]
                    elif isinstance(raw_items, (list, tuple)):
                        items_list = [str(i).strip() for i in raw_items]

                # Try to get item descriptions
                raw_desc = getattr(filing, "item_descriptions", None)
                if raw_desc:
                    if isinstance(raw_desc, str):
                        descriptions = [raw_desc]
                    elif isinstance(raw_desc, (list, tuple)):
                        descriptions = [str(d) for d in raw_desc]

                # Determine boolean flags from item numbers
                categories = set()
                for item in items_list:
                    # Normalize item number (e.g., "Item 2.02" -> "2.02")
                    item_num = item.replace("Item", "").strip()
                    cat = _ITEM_MAP.get(item_num)
                    if cat:
                        categories.add(cat)

                rows.append(
                    {
                        "filed_date": filed_date,
                        "event_date": event_date,
                        "items": items_list,
                        "item_descriptions": descriptions,
                        "has_earnings": "earnings" in categories,
                        "has_acquisition": "acquisition" in categories,
                        "has_officer_change": "officer_change" in categories,
                        "has_restatement": "restatement" in categories,
                        "has_material_agreement": "material_agreement"
                        in categories,
                        "has_delisting_notice": "delisting" in categories,
                        "has_bankruptcy": "bankruptcy" in categories,
                    }
                )
            except Exception:
                logger.debug("Failed to parse 8-K filing for %s", symbol)
                continue

        if not rows:
            return self._empty_df()

        df = pd.DataFrame(rows)
        # Ensure all columns present in correct order
        for col in _COLUMNS:
            if col not in df.columns:
                df[col] = False if col.startswith("has_") else None
        df = df[_COLUMNS]
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
