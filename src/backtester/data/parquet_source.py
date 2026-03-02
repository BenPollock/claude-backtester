"""Parquet file data source (Gap 8)."""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from backtester.data.sources.base import DataSource

logger = logging.getLogger(__name__)


class ParquetDataSource(DataSource):
    """Loads OHLCV data from Parquet files: {data_path}/{SYMBOL}.parquet

    Expected columns: Open, High, Low, Close, Volume
    """

    def __init__(self, data_path: str):
        self._data_path = Path(data_path)

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        path = self._data_path / f"{symbol.upper()}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No Parquet data file for {symbol} at {path}")

        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(df.index, name="Date")
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.DatetimeIndex(df.index.date, name="Date")

        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        df = df.loc[mask]

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Parquet for {symbol} missing columns: {missing}")

        return df[required].copy()
