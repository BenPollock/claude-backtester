"""Tests for analytics/report.py: print_report, export_activity_log_csv, _print_activity_log."""

import csv
from datetime import date
from io import StringIO

import pandas as pd
import pytest

from backtester.config import BacktestConfig
from backtester.portfolio.portfolio import Portfolio
from backtester.portfolio.order import Trade, TradeLogEntry
from backtester.result import BacktestResult
from backtester.types import Side
from backtester.analytics.report import (
    print_report,
    export_activity_log_csv,
    _print_activity_log,
)


# ── Fixtures ────────────────────────────────────────────────────────


def _minimal_config(**overrides) -> BacktestConfig:
    """Build a BacktestConfig with sensible defaults, overridable via kwargs."""
    defaults = dict(
        strategy_name="test_strategy",
        tickers=["AAPL"],
        benchmark="SPY",
        start_date=date(2020, 1, 2),
        end_date=date(2020, 12, 31),
        starting_cash=100_000.0,
        max_positions=10,
        max_alloc_pct=0.10,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _build_equity_history(start_value: float = 100_000.0,
                          days: int = 60,
                          daily_return: float = 0.0003) -> list[tuple[date, float]]:
    """Generate a simple equity history with consistent small daily returns."""
    dates = pd.bdate_range(start="2020-01-02", periods=days, freq="B")
    value = start_value
    history = []
    for d in dates:
        history.append((d.date(), value))
        value *= (1 + daily_return)
    return history


def _build_benchmark_equity(start_value: float = 100_000.0,
                            days: int = 60,
                            daily_return: float = 0.0002) -> list[tuple[date, float]]:
    """Generate benchmark equity data aligned with equity history dates."""
    dates = pd.bdate_range(start="2020-01-02", periods=days, freq="B")
    value = start_value
    history = []
    for d in dates:
        history.append((d.date(), value))
        value *= (1 + daily_return)
    return history


def _make_trade(pnl: float = 100.0, holding_days: int = 10,
                entry_date: date | None = None,
                exit_date: date | None = None) -> Trade:
    """Create a Trade with sensible defaults."""
    return Trade(
        symbol="AAPL",
        entry_date=entry_date or date(2020, 3, 2),
        exit_date=exit_date or date(2020, 3, 16),
        entry_price=150.0,
        exit_price=150.0 + pnl / 10,
        quantity=10,
        pnl=pnl,
        pnl_pct=pnl / 1500.0,
        holding_days=holding_days,
        fees_total=1.0,
    )


def _make_activity_entry(action: Side = Side.BUY,
                         avg_cost_basis: float | None = 150.0) -> TradeLogEntry:
    """Create a TradeLogEntry with sensible defaults."""
    return TradeLogEntry(
        date=date(2020, 3, 2),
        symbol="AAPL",
        action=action,
        quantity=10,
        price=150.0,
        value=1500.0,
        avg_cost_basis=avg_cost_basis,
        fees=1.0,
        slippage=0.05,
    )


def _result_with_trades_and_benchmark() -> BacktestResult:
    """Build a BacktestResult with trades, activity log, and benchmark data."""
    config = _minimal_config()
    portfolio = Portfolio(cash=100_000.0)
    portfolio.equity_history = _build_equity_history()

    # Add a winning and a losing trade for richer output
    portfolio.trade_log = [
        _make_trade(pnl=200.0, holding_days=10,
                    entry_date=date(2020, 2, 3), exit_date=date(2020, 2, 17)),
        _make_trade(pnl=-50.0, holding_days=5,
                    entry_date=date(2020, 3, 2), exit_date=date(2020, 3, 9)),
    ]
    portfolio.activity_log = [
        _make_activity_entry(Side.BUY, avg_cost_basis=None),
        _make_activity_entry(Side.SELL, avg_cost_basis=150.0),
    ]

    benchmark_equity = _build_benchmark_equity()
    return BacktestResult(config, portfolio, benchmark_equity=benchmark_equity)


# ── print_report tests ──────────────────────────────────────────────


class TestPrintReport:
    """Tests for print_report()."""

    def test_minimal_no_trades_no_benchmark(self, capsys):
        """print_report with no trades and no benchmark prints without error
        and returns a dict with expected metric keys."""
        config = _minimal_config()
        portfolio = Portfolio(cash=100_000.0)
        portfolio.equity_history = _build_equity_history()
        result = BacktestResult(config, portfolio, benchmark_equity=None)

        metrics = print_report(result)

        # Verify return type and essential keys
        assert isinstance(metrics, dict)
        for key in ("total_return", "cagr", "sharpe_ratio", "sortino_ratio",
                     "max_drawdown", "total_trades", "win_rate",
                     "profit_factor", "trade_expectancy", "exposure_time"):
            assert key in metrics, f"Missing key: {key}"

        # No benchmark-relative keys when benchmark is None
        assert "alpha" not in metrics
        assert "beta" not in metrics

        # Trade-level stats should reflect zero trades
        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0.0

        # Verify console output was produced
        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out
        assert "Strategy Performance" in captured.out
        # Should NOT have benchmark section
        assert "Benchmark Buy & Hold" not in captured.out
        assert "Benchmark-Relative" not in captured.out

    def test_with_trades_and_benchmark(self, capsys):
        """print_report with trades and benchmark includes benchmark-relative
        section and trade statistics in output."""
        result = _result_with_trades_and_benchmark()

        metrics = print_report(result)

        # Benchmark-relative metrics should be present
        assert "alpha" in metrics
        assert "beta" in metrics
        assert "information_ratio" in metrics
        assert "tracking_error" in metrics
        assert "up_capture" in metrics
        assert "down_capture" in metrics

        # Trade stats
        assert metrics["total_trades"] == 2

        # Console output should have benchmark and trades sections
        captured = capsys.readouterr()
        assert "Benchmark Buy & Hold" in captured.out
        assert "Benchmark-Relative" in captured.out
        assert "Alpha" in captured.out
        assert "Beta" in captured.out
        assert "Trades" in captured.out
        assert "Avg Trade PnL" in captured.out
        assert "Best Trade" in captured.out
        assert "Worst Trade" in captured.out
        assert "Avg Winner" in captured.out
        assert "Avg Loser" in captured.out
        assert "Payoff Ratio" in captured.out
        assert "Max Consec Win" in captured.out

    def test_output_includes_config_details(self, capsys):
        """print_report outputs the strategy name, tickers, period, etc."""
        config = _minimal_config(
            strategy_name="my_strat",
            tickers=["AAPL", "MSFT"],
            starting_cash=50_000.0,
        )
        portfolio = Portfolio(cash=50_000.0)
        portfolio.equity_history = _build_equity_history(start_value=50_000.0)
        result = BacktestResult(config, portfolio)

        print_report(result)

        captured = capsys.readouterr()
        assert "my_strat" in captured.out
        assert "AAPL, MSFT" in captured.out
        assert "$50,000.00" in captured.out

    def test_calmar_ratio_inf_display(self, capsys):
        """When equity only goes up (no drawdown), calmar ratio displays as inf."""
        config = _minimal_config()
        portfolio = Portfolio(cash=100_000.0)
        # Strictly increasing equity -> max_drawdown = 0 -> calmar = inf
        days = 60
        dates = pd.bdate_range(start="2020-01-02", periods=days, freq="B")
        value = 100_000.0
        for d in dates:
            portfolio.equity_history.append((d.date(), value))
            value += 100.0  # always going up

        result = BacktestResult(config, portfolio)
        metrics = print_report(result)

        captured = capsys.readouterr()
        # Calmar should be printed as "inf" since drawdown is 0
        assert "inf" in captured.out


# ── export_activity_log_csv tests ───────────────────────────────────


class TestExportActivityLogCsv:
    """Tests for export_activity_log_csv()."""

    def test_writes_csv_with_correct_headers(self, tmp_path):
        """CSV file has expected headers and correct number of data rows."""
        result = _result_with_trades_and_benchmark()
        filepath = str(tmp_path / "activity.csv")

        export_activity_log_csv(result, filepath)

        with open(filepath) as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert headers == [
                "date", "symbol", "action", "quantity", "price",
                "value", "avg_cost_basis", "fees", "slippage",
            ]
            rows = list(reader)
            assert len(rows) == 2  # two activity log entries

    def test_empty_activity_log_produces_header_only_csv(self, tmp_path):
        """With no activity, CSV should have headers but no data rows."""
        config = _minimal_config()
        portfolio = Portfolio(cash=100_000.0)
        portfolio.equity_history = _build_equity_history()
        result = BacktestResult(config, portfolio)
        filepath = str(tmp_path / "empty_log.csv")

        export_activity_log_csv(result, filepath)

        with open(filepath) as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert headers == [
                "date", "symbol", "action", "quantity", "price",
                "value", "avg_cost_basis", "fees", "slippage",
            ]
            rows = list(reader)
            assert len(rows) == 0

    def test_avg_cost_basis_none_written_as_empty_string(self, tmp_path):
        """When avg_cost_basis is None, the CSV cell should be empty."""
        config = _minimal_config()
        portfolio = Portfolio(cash=100_000.0)
        portfolio.equity_history = _build_equity_history()
        portfolio.activity_log = [_make_activity_entry(Side.BUY, avg_cost_basis=None)]
        result = BacktestResult(config, portfolio)
        filepath = str(tmp_path / "null_basis.csv")

        export_activity_log_csv(result, filepath)

        with open(filepath) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            row = next(reader)
            # avg_cost_basis is column index 6
            assert row[6] == ""

    def test_creates_parent_directories(self, tmp_path):
        """export_activity_log_csv creates intermediate directories as needed."""
        result = _result_with_trades_and_benchmark()
        filepath = str(tmp_path / "nested" / "dir" / "activity.csv")

        export_activity_log_csv(result, filepath)

        with open(filepath) as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert len(headers) == 9


# ── _print_activity_log tests ───────────────────────────────────────


class TestPrintActivityLog:
    """Tests for _print_activity_log() (internal helper)."""

    def test_handles_avg_cost_basis_none(self, capsys):
        """_print_activity_log prints N/A when avg_cost_basis is None."""
        config = _minimal_config()
        portfolio = Portfolio(cash=100_000.0)
        portfolio.equity_history = _build_equity_history()
        portfolio.activity_log = [_make_activity_entry(Side.BUY, avg_cost_basis=None)]
        result = BacktestResult(config, portfolio)

        _print_activity_log(result)

        captured = capsys.readouterr()
        assert "N/A" in captured.out

    def test_handles_avg_cost_basis_present(self, capsys):
        """_print_activity_log prints the cost basis value when present."""
        config = _minimal_config()
        portfolio = Portfolio(cash=100_000.0)
        portfolio.equity_history = _build_equity_history()
        portfolio.activity_log = [_make_activity_entry(Side.SELL, avg_cost_basis=149.50)]
        result = BacktestResult(config, portfolio)

        _print_activity_log(result)

        captured = capsys.readouterr()
        assert "149.50" in captured.out

    def test_empty_activity_log_produces_no_output(self, capsys):
        """_print_activity_log with no entries produces no output."""
        config = _minimal_config()
        portfolio = Portfolio(cash=100_000.0)
        portfolio.equity_history = _build_equity_history()
        result = BacktestResult(config, portfolio)

        _print_activity_log(result)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_activity_log_header_present(self, capsys):
        """_print_activity_log prints the Activity Log header and column names."""
        config = _minimal_config()
        portfolio = Portfolio(cash=100_000.0)
        portfolio.equity_history = _build_equity_history()
        portfolio.activity_log = [_make_activity_entry()]
        result = BacktestResult(config, portfolio)

        _print_activity_log(result)

        captured = capsys.readouterr()
        assert "Activity Log" in captured.out
        assert "Ticker" in captured.out
        assert "Action" in captured.out
        assert "Price" in captured.out
