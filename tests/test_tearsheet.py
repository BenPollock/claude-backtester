"""Tests for HTML tearsheet generation."""

import base64
import os
import tempfile
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from backtester.analytics.tearsheet import (
    generate_tearsheet,
    _render_chart_as_base64,
    _generate_monthly_returns_html,
    _generate_trade_table_html,
)
from backtester.config import BacktestConfig
from backtester.portfolio.portfolio import Portfolio
from backtester.portfolio.order import Trade
from backtester.result import BacktestResult


# ── Helpers ──────────────────────────────────────────────────────────


def _make_portfolio_with_equity(
    cash: float = 100_000.0,
    days: int = 252,
    start: str = "2020-01-02",
    daily_return: float = 0.0005,
    trades: list[Trade] | None = None,
) -> Portfolio:
    """Create a Portfolio with synthetic equity history and optional trades."""
    portfolio = Portfolio(cash=cash)
    dates = pd.bdate_range(start=start, periods=days)
    rng = np.random.default_rng(42)
    value = cash
    for d in dates:
        value *= 1 + daily_return + rng.normal(0, 0.01)
        portfolio.equity_history.append((d.date(), value))
    portfolio.cash = value
    if trades:
        portfolio.trade_log = trades
    return portfolio


def _make_sample_trades(n: int = 10) -> list[Trade]:
    """Create a list of synthetic Trade objects."""
    rng = np.random.default_rng(99)
    trades = []
    base = date(2020, 3, 1)
    for i in range(n):
        entry = date(2020, 3 + i % 10, 1 + i)
        exit_d = date(2020, 3 + i % 10, 10 + i)
        entry_price = 100.0 + rng.uniform(-5, 5)
        pnl = rng.uniform(-200, 400)
        pnl_pct = pnl / (entry_price * 10)
        exit_price = entry_price * (1 + pnl_pct)
        trades.append(
            Trade(
                symbol="TEST",
                entry_date=entry,
                exit_date=exit_d,
                entry_price=round(entry_price, 2),
                exit_price=round(exit_price, 2),
                quantity=10,
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 4),
                holding_days=(exit_d - entry).days,
                fees_total=0.10,
            )
        )
    return trades


def _make_config(**overrides) -> BacktestConfig:
    """Create a minimal BacktestConfig with optional overrides."""
    defaults = dict(
        strategy_name="sma_crossover",
        tickers=["TEST"],
        benchmark="TEST",
        start_date=date(2020, 1, 2),
        end_date=date(2020, 12, 31),
        starting_cash=100_000.0,
        max_positions=10,
        max_alloc_pct=0.10,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _make_result(
    days: int = 252,
    trades: list[Trade] | None = None,
    include_benchmark: bool = False,
) -> BacktestResult:
    """Create a minimal BacktestResult for testing."""
    config = _make_config()
    portfolio = _make_portfolio_with_equity(
        cash=config.starting_cash, days=days, trades=trades or []
    )

    benchmark_equity = None
    if include_benchmark and portfolio.equity_history:
        # Mirror equity history as benchmark with slight offset
        benchmark_equity = [
            (d, v * 0.98) for d, v in portfolio.equity_history
        ]

    return BacktestResult(
        config=config,
        portfolio=portfolio,
        benchmark_equity=benchmark_equity,
    )


# ── Test fixtures ────────────────────────────────────────────────────


@pytest.fixture
def tmp_html(tmp_path):
    """Return a temporary HTML file path, cleaned up after test."""
    return str(tmp_path / "test_tearsheet.html")


@pytest.fixture
def result_with_trades():
    """BacktestResult with synthetic trades."""
    return _make_result(days=252, trades=_make_sample_trades(10))


@pytest.fixture
def result_no_trades():
    """BacktestResult with no trades."""
    return _make_result(days=252, trades=[])


@pytest.fixture
def result_short_period():
    """BacktestResult with a very short backtest (10 days)."""
    config = _make_config(
        start_date=date(2020, 1, 2),
        end_date=date(2020, 1, 15),
    )
    portfolio = _make_portfolio_with_equity(cash=100_000.0, days=10)
    return BacktestResult(config=config, portfolio=portfolio)


# ── Tests: generate_tearsheet ────────────────────────────────────────


class TestGenerateTearsheet:

    def test_produces_html_file(self, result_with_trades, tmp_html):
        path = generate_tearsheet(result_with_trades, output_path=tmp_html)
        assert os.path.exists(path)
        assert path == tmp_html

    def test_html_is_valid(self, result_with_trades, tmp_html):
        generate_tearsheet(result_with_trades, output_path=tmp_html)
        with open(tmp_html, "r", encoding="utf-8") as f:
            content = f.read()
        assert content.startswith("<!DOCTYPE html>")
        assert "</html>" in content

    def test_contains_key_sections(self, result_with_trades, tmp_html):
        generate_tearsheet(result_with_trades, output_path=tmp_html)
        with open(tmp_html, "r", encoding="utf-8") as f:
            content = f.read()
        # Metrics table
        assert "Key Performance Metrics" in content
        assert "CAGR" in content
        assert "Sharpe Ratio" in content
        assert "Max Drawdown" in content
        # Charts
        assert "Equity Curve" in content
        assert "data:image/png;base64," in content
        # Monthly returns
        assert "Monthly Returns" in content
        # Trade stats
        assert "Trade Statistics" in content
        # Trade list
        assert "Recent Trades" in content

    def test_html_is_self_contained(self, result_with_trades, tmp_html):
        """Verify no external resource references (CSS, JS, images)."""
        generate_tearsheet(result_with_trades, output_path=tmp_html)
        with open(tmp_html, "r", encoding="utf-8") as f:
            content = f.read()
        # No external stylesheet links
        assert '<link rel="stylesheet"' not in content
        # No external scripts
        assert "<script src=" not in content
        # No external image sources (all images should be base64)
        assert 'src="http' not in content

    def test_returns_output_path(self, result_with_trades, tmp_html):
        path = generate_tearsheet(result_with_trades, output_path=tmp_html)
        assert path == tmp_html

    def test_works_with_empty_trades(self, result_no_trades, tmp_html):
        path = generate_tearsheet(result_no_trades, output_path=tmp_html)
        assert os.path.exists(path)
        with open(tmp_html, "r", encoding="utf-8") as f:
            content = f.read()
        assert "No trades executed" in content

    def test_works_with_short_period(self, result_short_period, tmp_html):
        path = generate_tearsheet(result_short_period, output_path=tmp_html)
        assert os.path.exists(path)
        with open(tmp_html, "r", encoding="utf-8") as f:
            content = f.read()
        # Should not contain rolling metrics section (too few data points)
        assert "Rolling 12-Month Metrics" not in content

    def test_works_with_benchmark(self, tmp_html):
        result = _make_result(days=252, trades=_make_sample_trades(5),
                              include_benchmark=True)
        path = generate_tearsheet(result, output_path=tmp_html)
        assert os.path.exists(path)
        with open(tmp_html, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Benchmark" in content

    def test_contains_strategy_name(self, result_with_trades, tmp_html):
        generate_tearsheet(result_with_trades, output_path=tmp_html)
        with open(tmp_html, "r", encoding="utf-8") as f:
            content = f.read()
        assert "sma_crossover" in content

    def test_output_to_custom_path(self, result_with_trades, tmp_path):
        custom_path = str(tmp_path / "custom_dir" / "report.html")
        os.makedirs(os.path.dirname(custom_path), exist_ok=True)
        path = generate_tearsheet(result_with_trades, output_path=custom_path)
        assert os.path.exists(path)
        assert path == custom_path


# ── Tests: _render_chart_as_base64 ──────────────────────────────────


class TestRenderChartAsBase64:

    def test_returns_valid_base64(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        result = _render_chart_as_base64(fig)
        # Should be a non-empty string
        assert isinstance(result, str)
        assert len(result) > 100

    def test_decodable_to_png(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        result = _render_chart_as_base64(fig)
        # Decode and verify PNG magic bytes
        raw = base64.b64decode(result)
        assert raw[:4] == b"\x89PNG"

    def test_closes_figure(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig_num = fig.number
        _render_chart_as_base64(fig)
        # Figure should be closed
        assert fig_num not in plt.get_fignums()


# ── Tests: _generate_monthly_returns_html ───────────────────────────


class TestGenerateMonthlyReturnsHtml:

    def test_produces_table(self):
        dates = pd.bdate_range("2020-01-02", periods=252)
        values = 100_000 * np.cumprod(1 + np.random.default_rng(42).normal(0.0003, 0.01, 252))
        equity = pd.Series(values, index=dates)
        html = _generate_monthly_returns_html(equity)
        assert "<table" in html
        assert "Jan" in html
        assert "2020" in html

    def test_contains_ytd_column(self):
        dates = pd.bdate_range("2020-01-02", periods=252)
        values = 100_000 * np.cumprod(1 + np.random.default_rng(42).normal(0.0003, 0.01, 252))
        equity = pd.Series(values, index=dates)
        html = _generate_monthly_returns_html(equity)
        assert "YTD" in html

    def test_handles_short_series(self):
        dates = pd.bdate_range("2020-01-02", periods=3)
        values = [100_000, 100_100, 100_200]
        equity = pd.Series(values, index=dates)
        html = _generate_monthly_returns_html(equity)
        # Should produce some output without error
        assert isinstance(html, str)
        assert len(html) > 0

    def test_insufficient_data(self):
        equity = pd.Series([100_000.0], index=pd.DatetimeIndex([date(2020, 1, 2)]))
        html = _generate_monthly_returns_html(equity)
        assert "Insufficient data" in html


# ── Tests: _generate_trade_table_html ───────────────────────────────


class TestGenerateTradeTableHtml:

    def test_renders_trades(self):
        trades = _make_sample_trades(5)
        html = _generate_trade_table_html(trades)
        assert "<table" in html
        assert "Entry Date" in html
        assert "TEST" in html

    def test_limits_rows(self):
        trades = _make_sample_trades(10)
        html = _generate_trade_table_html(trades, max_rows=3)
        # Should show "last 3 of 10" note
        assert "3 of 10" in html

    def test_empty_trades(self):
        html = _generate_trade_table_html([])
        assert "No trades executed" in html

    def test_shows_pnl_classes(self):
        trades = _make_sample_trades(5)
        html = _generate_trade_table_html(trades)
        # Should contain CSS classes for coloring
        assert "positive" in html or "negative" in html
