"""CSV file data source (Gap 8)."""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from backtester.data.sources.base import DataSource

logger = logging.getLogger(__name__)


class CSVDataSource(DataSource):
    """Loads OHLCV data from CSV files: {data_path}/{SYMBOL}.csv

    Expected columns: Date, Open, High, Low, Close, Volume
    """

    def __init__(self, data_path: str):
        self._data_path = Path(data_path)

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        path = self._data_path / f"{symbol.upper()}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No CSV data file for {symbol} at {path}")

        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        df.index = pd.DatetimeIndex(df.index.date, name="Date")

        # Filter to date range
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        df = df.loc[mask]

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"CSV for {symbol} missing columns: {missing}")

        return df[required].copy()
