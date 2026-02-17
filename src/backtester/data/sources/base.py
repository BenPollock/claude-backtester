"""Abstract base class for data sources."""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DataSource(ABC):
    """Interface for fetching historical OHLCV data."""

    @abstractmethod
    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol.

        Returns DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (date-only, timezone-naive)
        """
        ...
