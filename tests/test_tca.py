"""Tests for Transaction Cost Analysis, Turnover, and Capacity estimation."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtester.analytics.tca import (
    compute_turnover,
    compute_cost_attribution,
    estimate_capacity,
)
from backtester.portfolio.order import Trade


def _make_equity(start="2020-01-02", periods=252, start_val=100_000, end_val=110_000):
    dates = pd.bdate_range(start=start, periods=periods)
    vals = np.linspace(start_val, end_val, periods)
    return pd.Series(vals, index=pd.DatetimeIndex(dates.date, name="Date"))


def _make_trade(symbol="SPY", entry_date=date(2020, 3, 1), exit_date=date(2020, 4, 1),
                entry_price=300.0, exit_price=310.0, quantity=100, pnl=1000.0,
                fees_total=10.0):
    return Trade(
        symbol=symbol,
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        pnl=pnl,
        pnl_pct=pnl / (entry_price * quantity),
        holding_days=(exit_date - entry_date).days,
        fees_total=fees_total,
    )


def _make_price_df(dates_list, close_prices, volumes):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates_list], name="Date")
    return pd.DataFrame({
        "Open": close_prices,
        "High": [p * 1.01 for p in close_prices],
        "Low": [p * 0.99 for p in close_prices],
        "Close": close_prices,
        "Volume": volumes,
    }, index=idx)


class TestComputeTurnover:
    def test_basic_turnover(self):
        equity = _make_equity(periods=252)
        trades = [_make_trade(quantity=100, entry_price=300.0)]
        result = compute_turnover(trades, equity)
        # total_traded = 100 * 300 = 30000
        avg_equity = equity.mean()
        days = (equity.index[-1] - equity.index[0]).days
        years = days / 365.25
        expected = 30000.0 / avg_equity / years
        assert result == pytest.approx(expected, rel=1e-6)

    def test_no_trades_returns_zero(self):
        equity = _make_equity()
        assert compute_turnover([], equity) == 0.0

    def test_short_equity_returns_zero(self):
        equity = pd.Series([100_000.0], index=[date(2020, 1, 2)])
        trades = [_make_trade()]
        assert compute_turnover(trades, equity) == 0.0

    def test_zero_avg_equity_returns_zero(self):
        dates = pd.bdate_range("2020-01-02", periods=10)
        equity = pd.Series(np.zeros(10), index=pd.DatetimeIndex(dates.date, name="Date"))
        trades = [_make_trade()]
        assert compute_turnover(trades, equity) == 0.0

    def test_multiple_trades_increases_turnover(self):
        equity = _make_equity()
        one_trade = [_make_trade(quantity=100, entry_price=300.0)]
        two_trades = [
            _make_trade(quantity=100, entry_price=300.0),
            _make_trade(quantity=200, entry_price=300.0),
        ]
        t1 = compute_turnover(one_trade, equity)
        t2 = compute_turnover(two_trades, equity)
        assert t2 > t1


class TestComputeCostAttribution:
    def test_basic_attribution(self):
        equity = _make_equity(start_val=100_000, end_val=110_000)
        trades = [_make_trade(fees_total=50.0), _make_trade(fees_total=30.0)]
        result = compute_cost_attribution(trades, equity)

        assert result["total_fees"] == 80.0
        assert result["cost_pct_equity"] > 0
        assert result["cost_pct_return"] > 0
        # 80 / ~105000
        assert result["cost_pct_equity"] == pytest.approx(80.0 / equity.mean(), rel=1e-6)
        # 80 / 10000
        assert result["cost_pct_return"] == pytest.approx(80.0 / 10_000.0, rel=1e-6)

    def test_no_trades(self):
        equity = _make_equity()
        result = compute_cost_attribution([], equity)
        assert result == {"total_fees": 0.0, "cost_pct_equity": 0.0, "cost_pct_return": 0.0}

    def test_zero_return(self):
        equity = _make_equity(start_val=100_000, end_val=100_000)
        trades = [_make_trade(fees_total=50.0)]
        result = compute_cost_attribution(trades, equity)
        assert result["total_fees"] == 50.0
        assert result["cost_pct_return"] == 0.0

    def test_negative_return(self):
        equity = _make_equity(start_val=100_000, end_val=90_000)
        trades = [_make_trade(fees_total=50.0)]
        result = compute_cost_attribution(trades, equity)
        assert result["total_fees"] == 50.0
        # cost_pct_return uses abs(total_return_val)
        assert result["cost_pct_return"] == pytest.approx(50.0 / 10_000.0, rel=1e-6)


class TestEstimateCapacity:
    def test_basic_capacity(self):
        trade = _make_trade(symbol="SPY", entry_date=date(2020, 3, 2))
        price_df = _make_price_df(
            [date(2020, 3, 2)], [300.0], [1_000_000]
        )
        result = estimate_capacity([trade], {"SPY": price_df}, max_volume_pct=0.01)
        # max_shares = int(1_000_000 * 0.01) = 10000
        # max_value = 10000 * 300 = 3_000_000
        assert result == 3_000_000

    def test_no_trades(self):
        assert estimate_capacity([], {"SPY": pd.DataFrame()}) == 0.0

    def test_no_price_data(self):
        trade = _make_trade()
        assert estimate_capacity([trade], {}) == 0.0

    def test_symbol_not_in_price_data(self):
        trade = _make_trade(symbol="AAPL")
        price_df = _make_price_df([date(2020, 3, 2)], [300.0], [1_000_000])
        result = estimate_capacity([trade], {"SPY": price_df})
        assert result == 0.0

    def test_date_not_in_index(self):
        trade = _make_trade(symbol="SPY", entry_date=date(2020, 6, 1))
        price_df = _make_price_df([date(2020, 3, 2)], [300.0], [1_000_000])
        result = estimate_capacity([trade], {"SPY": price_df})
        assert result == 0.0

    def test_minimum_capacity_selected(self):
        # Two trades, second has lower volume => lower capacity
        trade1 = _make_trade(symbol="SPY", entry_date=date(2020, 3, 2))
        trade2 = _make_trade(symbol="AAPL", entry_date=date(2020, 3, 2))
        spy_df = _make_price_df([date(2020, 3, 2)], [300.0], [1_000_000])
        aapl_df = _make_price_df([date(2020, 3, 2)], [200.0], [100_000])
        result = estimate_capacity(
            [trade1, trade2],
            {"SPY": spy_df, "AAPL": aapl_df},
            max_volume_pct=0.01,
        )
        # SPY: int(1M*0.01)*300 = 3M, AAPL: int(100k*0.01)*200 = 200k
        assert result == 200_000

    def test_zero_volume(self):
        trade = _make_trade(symbol="SPY", entry_date=date(2020, 3, 2))
        price_df = _make_price_df([date(2020, 3, 2)], [300.0], [0])
        result = estimate_capacity([trade], {"SPY": price_df})
        assert result == 0.0

    def test_custom_max_volume_pct(self):
        trade = _make_trade(symbol="SPY", entry_date=date(2020, 3, 2))
        price_df = _make_price_df([date(2020, 3, 2)], [300.0], [1_000_000])
        result = estimate_capacity([trade], {"SPY": price_df}, max_volume_pct=0.05)
        # max_shares = int(1M * 0.05) = 50000, max_value = 50000 * 300 = 15M
        assert result == 15_000_000
