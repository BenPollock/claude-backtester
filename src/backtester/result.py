"""BacktestResult container for backtest outputs."""

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from backtester.config import BacktestConfig
from backtester.portfolio.portfolio import Portfolio

logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtest outputs."""

    def __init__(self, config: BacktestConfig, portfolio: Portfolio,
                 benchmark_equity: list[tuple[date, float]] | None = None,
                 benchmark_prices: pd.Series | None = None,
                 universe_data: dict[str, pd.DataFrame] | None = None):
        self.config = config
        self.portfolio = portfolio
        self.benchmark_equity = benchmark_equity
        self.benchmark_prices = benchmark_prices
        self.universe_data = universe_data

    @property
    def equity_series(self) -> pd.Series:
        dates, values = zip(*self.portfolio.equity_history)
        return pd.Series(values, index=pd.DatetimeIndex(dates, name="Date"), name="Equity")

    @property
    def benchmark_series(self) -> pd.Series | None:
        if not self.benchmark_equity:
            return None
        dates, values = zip(*self.benchmark_equity)
        return pd.Series(values, index=pd.DatetimeIndex(dates, name="Date"), name="Benchmark")

    @property
    def trades(self):
        return self.portfolio.trade_log

    @property
    def activity_log(self):
        return self.portfolio.activity_log

    # --- Gap 30: Result persistence ---

    def save(self, path: str) -> None:
        """Save backtest results to a directory.

        Creates: config.json, metrics.json, equity.parquet,
                 trades.parquet, benchmark.parquet (if available).
        """
        from backtester.analytics.metrics import compute_all_metrics

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        # Config as JSON
        config_dict = {}
        for field_name in self.config.__dataclass_fields__:
            val = getattr(self.config, field_name)
            if isinstance(val, date):
                val = val.isoformat()
            elif hasattr(val, '__dataclass_fields__'):
                val = {k: getattr(val, k) for k in val.__dataclass_fields__}
            elif isinstance(val, (list, dict, str, int, float, bool, type(None))):
                pass
            else:
                val = str(val)
            config_dict[field_name] = val
        (out / "config.json").write_text(json.dumps(config_dict, indent=2, default=str))

        # Metrics
        equity = self.equity_series
        bm = self.benchmark_series
        metrics = compute_all_metrics(equity, self.trades, benchmark_series=bm)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

        # Equity curve
        equity.to_frame().to_parquet(out / "equity.parquet")

        # Trades
        if self.trades:
            trade_dicts = []
            for t in self.trades:
                trade_dicts.append({
                    "symbol": t.symbol,
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "holding_days": t.holding_days,
                    "fees_total": t.fees_total,
                })
            pd.DataFrame(trade_dicts).to_parquet(out / "trades.parquet")

        # Benchmark
        if bm is not None:
            bm.to_frame().to_parquet(out / "benchmark.parquet")

        logger.info(f"Results saved to {out}")

    @classmethod
    def load(cls, path: str) -> "BacktestResult":
        """Load a saved BacktestResult (lightweight — no portfolio state)."""
        p = Path(path)

        config_dict = json.loads((p / "config.json").read_text())
        metrics = json.loads((p / "metrics.json").read_text())

        equity_df = pd.read_parquet(p / "equity.parquet")
        equity_series = equity_df.iloc[:, 0]

        benchmark_series = None
        bm_path = p / "benchmark.parquet"
        if bm_path.exists():
            bm_df = pd.read_parquet(bm_path)
            benchmark_series = bm_df.iloc[:, 0]

        # Create a lightweight result container
        result = cls.__new__(cls)
        result.config = None
        result.portfolio = None
        result.benchmark_equity = None
        result.benchmark_prices = None
        result.universe_data = None
        result._loaded_equity = equity_series
        result._loaded_benchmark = benchmark_series
        result._loaded_metrics = metrics
        result._loaded_config = config_dict
        return result

    @staticmethod
    def compare(paths: list[str]) -> pd.DataFrame:
        """Load multiple saved results and produce a side-by-side metrics table."""
        rows = []
        for path in paths:
            p = Path(path)
            metrics = json.loads((p / "metrics.json").read_text())
            config = json.loads((p / "config.json").read_text())
            row = {"path": path, "strategy": config.get("strategy_name", "")}
            row.update(metrics)
            rows.append(row)
        return pd.DataFrame(rows)
