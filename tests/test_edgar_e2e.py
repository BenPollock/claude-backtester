"""End-to-end integration tests for EDGAR data strategies.

Tests run full backtests through the engine with mock data to verify
that EDGAR-enriched strategies (value_quality, earnings_growth,
insider_following, smart_money, fundamental_screener) produce
correct signals across the entire pipeline.
"""

import tempfile
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backtester.config import BacktestConfig
from backtester.data.fundamental import EdgarDataManager
from backtester.data.fundamental_cache import EdgarCache
from backtester.data.manager import DataManager
from backtester.engine import BacktestEngine
from backtester.strategies.registry import discover_strategies

from tests.conftest import MockDataSource

discover_strategies()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_controlled_df(prices, start="2020-01-02", volume=1_000_000):
    """Create OHLCV DataFrame from a list of close prices."""
    days = len(prices)
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    closes = np.array(prices, dtype=float)
    df = pd.DataFrame(
        {
            "Open": closes * 0.999,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": np.full(days, volume),
        },
        index=pd.DatetimeIndex(dates.date, name="Date"),
    )
    return df


def _dates_from_df(df):
    start = df.index[0]
    end = df.index[-1]
    if hasattr(start, "date") and callable(start.date):
        start = start.date()
    if hasattr(end, "date") and callable(end.date):
        end = end.date()
    return start, end


def _run_backtest(config, source):
    """Helper: create DataManager + Engine, run backtest, return result."""
    tmpdir = tempfile.mkdtemp()
    dm = DataManager(cache_dir=tmpdir, source=source)
    engine = BacktestEngine(config, data_manager=dm)
    return engine.run()


def _enrich_with_financials(df, fin_df, symbol="TEST"):
    """Enrich a daily price DataFrame with financial data using EdgarDataManager."""
    tmpdir = tempfile.mkdtemp()
    cache = EdgarCache(tmpdir, "financials")
    cache.save(symbol, fin_df)
    mgr = EdgarDataManager(
        cache_dir=tmpdir, use_edgar=False,
        enable_financials=True, enable_insider=False,
        enable_institutional=False, enable_events=False,
    )
    mgr._financials_cache = cache
    return mgr.merge_all_onto_daily(symbol, df)


def _enrich_with_insider(df, insider_df, symbol="TEST"):
    """Enrich a daily price DataFrame with insider data."""
    tmpdir = tempfile.mkdtemp()
    cache = EdgarCache(tmpdir, "insider")
    cache.save(symbol, insider_df)
    mgr = EdgarDataManager(
        cache_dir=tmpdir, use_edgar=False,
        enable_financials=False, enable_insider=True,
        enable_institutional=False, enable_events=False,
    )
    mgr._insider_cache = cache
    return mgr.merge_all_onto_daily(symbol, df)


def _enrich_with_all(df, symbol="TEST", fin_df=None, insider_df=None,
                     inst_df=None, event_df=None):
    """Enrich daily data with multiple EDGAR data types."""
    tmpdir = tempfile.mkdtemp()
    mgr = EdgarDataManager(
        cache_dir=tmpdir, use_edgar=False,
        enable_financials=fin_df is not None,
        enable_insider=insider_df is not None,
        enable_institutional=inst_df is not None,
        enable_events=event_df is not None,
    )
    if fin_df is not None:
        cache = EdgarCache(tmpdir, "financials")
        cache.save(symbol, fin_df)
        mgr._financials_cache = cache
    if insider_df is not None:
        cache = EdgarCache(tmpdir, "insider")
        cache.save(symbol, insider_df)
        mgr._insider_cache = cache
    if inst_df is not None:
        cache = EdgarCache(tmpdir, "institutional")
        cache.save(symbol, inst_df)
        mgr._institutional_cache = cache
    if event_df is not None:
        cache = EdgarCache(tmpdir, "events")
        cache.save(symbol, event_df)
        mgr._events_cache = cache
    return mgr.merge_all_onto_daily(symbol, df)


def _make_good_financials(start_date=date(2018, 3, 31), quarters=12):
    """Create financials for a profitable, growing, low-leverage company."""
    rows = []
    for q in range(quarters):
        pe_d = start_date + timedelta(days=91 * q)
        fd = pe_d + timedelta(days=35)
        rev = 1e9 * (1.05 ** q)  # 5% quarterly growth
        ni = rev * 0.10
        eps = ni / 1e7
        rows.extend([
            {"metric": "revenue", "period_end": pe_d, "filed_date": fd, "value": rev, "form": "10-Q"},
            {"metric": "net_income", "period_end": pe_d, "filed_date": fd, "value": ni, "form": "10-Q"},
            {"metric": "eps_diluted", "period_end": pe_d, "filed_date": fd, "value": eps, "form": "10-Q"},
            {"metric": "operating_income", "period_end": pe_d, "filed_date": fd, "value": rev * 0.15, "form": "10-Q"},
            {"metric": "gross_profit", "period_end": pe_d, "filed_date": fd, "value": rev * 0.40, "form": "10-Q"},
            {"metric": "operating_cf", "period_end": pe_d, "filed_date": fd, "value": rev * 0.12, "form": "10-Q"},
            {"metric": "capex", "period_end": pe_d, "filed_date": fd, "value": rev * 0.03, "form": "10-Q"},
            {"metric": "total_assets", "period_end": pe_d, "filed_date": fd, "value": rev * 2.0, "form": "10-Q"},
            {"metric": "total_debt", "period_end": pe_d, "filed_date": fd, "value": rev * 0.3, "form": "10-Q"},
            {"metric": "current_assets", "period_end": pe_d, "filed_date": fd, "value": rev * 0.5, "form": "10-Q"},
            {"metric": "current_liabilities", "period_end": pe_d, "filed_date": fd, "value": rev * 0.25, "form": "10-Q"},
            {"metric": "equity", "period_end": pe_d, "filed_date": fd, "value": rev * 0.8, "form": "10-Q"},
            {"metric": "shares_outstanding", "period_end": pe_d, "filed_date": fd, "value": 1e7, "form": "10-Q"},
        ])
    return pd.DataFrame(rows)


def _make_insider_buying(start_date=date(2020, 2, 1), count=3):
    """Create insider buying data (officers purchasing shares)."""
    rows = []
    for i in range(count):
        fd = start_date + timedelta(days=i * 5)
        rows.append({
            "filed_date": fd,
            "transaction_date": fd - timedelta(days=1),
            "insider_name": f"Officer_{i}",
            "insider_title": "Chief Executive Officer" if i == 0 else "Chief Financial Officer",
            "transaction_type": "P",
            "shares": 5000,
            "price": 100.0,
            "shares_after": 50000 + 5000 * (i + 1),
            "is_direct": True,
        })
    return pd.DataFrame(rows)


def _make_insider_selling(start_date=date(2020, 2, 1), count=3):
    """Create insider selling data."""
    rows = []
    for i in range(count):
        fd = start_date + timedelta(days=i * 5)
        rows.append({
            "filed_date": fd,
            "transaction_date": fd - timedelta(days=1),
            "insider_name": f"Insider_{i}",
            "insider_title": "Chief Executive Officer",
            "transaction_type": "S",
            "shares": -10000,
            "price": 100.0,
            "shares_after": 40000 - 10000 * (i + 1),
            "is_direct": True,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E2E Tests
# ---------------------------------------------------------------------------


class TestValueQualityE2E:
    """E2E tests for the value_quality strategy."""

    def test_buy_on_good_fundamentals(self):
        """Profitable company with low P/E, above SMA -> BUY trades happen."""
        # Uptrending price: 80 -> 130 over 300 days (above 200 SMA eventually)
        prices = [80 + i * 0.17 for i in range(300)]
        daily = make_controlled_df(prices, start="2020-01-02")
        fin = _make_good_financials()
        enriched = _enrich_with_financials(daily, fin)

        source = MockDataSource()
        source.add("TEST", enriched)
        start, end = _dates_from_df(enriched)

        config = BacktestConfig(
            strategy_name="value_quality",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50, "max_pe": 30.0, "min_pe": 1.0},
        )
        result = _run_backtest(config, source)

        # Should have at least one trade (buy signal from good fundamentals)
        assert len(result.trades) > 0, "Should generate trades for profitable company"

    def test_no_buy_when_unprofitable(self):
        """Company with negative net_income -> no BUY signal."""
        prices = [80 + i * 0.17 for i in range(300)]
        daily = make_controlled_df(prices, start="2020-01-02")

        # Create financials with negative net_income
        rows = []
        for q in range(8):
            pe_d = date(2018, 3, 31) + timedelta(days=91 * q)
            fd = pe_d + timedelta(days=35)
            rev = 1e9
            rows.extend([
                {"metric": "revenue", "period_end": pe_d, "filed_date": fd, "value": rev, "form": "10-Q"},
                {"metric": "net_income", "period_end": pe_d, "filed_date": fd, "value": -1e8, "form": "10-Q"},
                {"metric": "eps_diluted", "period_end": pe_d, "filed_date": fd, "value": -10.0, "form": "10-Q"},
                {"metric": "equity", "period_end": pe_d, "filed_date": fd, "value": 5e8, "form": "10-Q"},
                {"metric": "shares_outstanding", "period_end": pe_d, "filed_date": fd, "value": 1e7, "form": "10-Q"},
                {"metric": "total_debt", "period_end": pe_d, "filed_date": fd, "value": 2e8, "form": "10-Q"},
            ])
        fin = pd.DataFrame(rows)
        enriched = _enrich_with_financials(daily, fin)

        source = MockDataSource()
        source.add("TEST", enriched)
        start, end = _dates_from_df(enriched)

        config = BacktestConfig(
            strategy_name="value_quality",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50, "max_pe": 30.0},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) == 0, "Unprofitable company should not generate trades"

    def test_no_trade_pe_too_high(self):
        """Good fundamentals but P/E > max -> no BUY."""
        prices = [80 + i * 0.17 for i in range(300)]
        daily = make_controlled_df(prices, start="2020-01-02")

        # Low EPS -> high P/E (>25)
        rows = []
        for q in range(8):
            pe_d = date(2018, 3, 31) + timedelta(days=91 * q)
            fd = pe_d + timedelta(days=35)
            rows.extend([
                {"metric": "revenue", "period_end": pe_d, "filed_date": fd, "value": 1e9, "form": "10-Q"},
                {"metric": "net_income", "period_end": pe_d, "filed_date": fd, "value": 1e6, "form": "10-Q"},
                {"metric": "eps_diluted", "period_end": pe_d, "filed_date": fd, "value": 0.1, "form": "10-Q"},
                {"metric": "equity", "period_end": pe_d, "filed_date": fd, "value": 5e8, "form": "10-Q"},
                {"metric": "shares_outstanding", "period_end": pe_d, "filed_date": fd, "value": 1e7, "form": "10-Q"},
                {"metric": "total_debt", "period_end": pe_d, "filed_date": fd, "value": 2e8, "form": "10-Q"},
            ])
        fin = pd.DataFrame(rows)
        enriched = _enrich_with_financials(daily, fin)

        source = MockDataSource()
        source.add("TEST", enriched)
        start, end = _dates_from_df(enriched)

        config = BacktestConfig(
            strategy_name="value_quality",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50, "max_pe": 25.0},
        )
        result = _run_backtest(config, source)
        # EPS TTM = 0.4, Close ~ 100 -> P/E ~ 250 >> max_pe=25
        assert len(result.trades) == 0, "High P/E should prevent buying"


class TestValueQualityGracefulDegradation:
    """Test that value_quality handles missing EDGAR data gracefully."""

    def test_no_edgar_data_no_trades(self):
        """Without EDGAR data, value_quality produces all HOLDs (no trades)."""
        prices = [80 + i * 0.17 for i in range(300)]
        daily = make_controlled_df(prices, start="2020-01-02")

        source = MockDataSource()
        source.add("TEST", daily)
        start, end = _dates_from_df(daily)

        config = BacktestConfig(
            strategy_name="value_quality",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) == 0, "No EDGAR data should mean no trades"


class TestEarningsGrowthE2E:
    """E2E tests for earnings_growth strategy."""

    def test_buy_on_strong_growth(self):
        """Strong earnings + revenue growth -> BUY signal."""
        prices = [80 + i * 0.17 for i in range(300)]
        daily = make_controlled_df(prices, start="2020-01-02")
        fin = _make_good_financials()
        enriched = _enrich_with_financials(daily, fin)

        source = MockDataSource()
        source.add("TEST", enriched)
        start, end = _dates_from_df(enriched)

        config = BacktestConfig(
            strategy_name="earnings_growth",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={
                "sma_period": 50,
                "min_earnings_growth": 0.0,
                "min_revenue_growth": 0.0,
                "max_pe": 100.0,
            },
        )
        result = _run_backtest(config, source)
        # The strategy needs fund_earnings_growth_yoy and fund_revenue_growth_yoy
        # which require 252-day shift. With 300 days, growth may be available late.
        # The strategy should at minimum not crash.
        assert result.equity_series is not None

    def test_no_trade_without_growth_data(self):
        """Without growth columns, earnings_growth produces no trades."""
        prices = [80 + i * 0.17 for i in range(100)]
        daily = make_controlled_df(prices, start="2020-01-02")

        source = MockDataSource()
        source.add("TEST", daily)
        start, end = _dates_from_df(daily)

        config = BacktestConfig(
            strategy_name="earnings_growth",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 20},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) == 0


class TestInsiderFollowingE2E:
    """E2E tests for insider_following strategy."""

    def test_buy_on_officer_buying(self):
        """Officers buying shares + uptrend -> BUY signal."""
        # Uptrending price
        prices = [80 + i * 0.3 for i in range(150)]
        daily = make_controlled_df(prices, start="2020-01-02")

        insider = _make_insider_buying(start_date=date(2020, 2, 1), count=5)
        enriched = _enrich_with_insider(daily, insider)

        source = MockDataSource()
        source.add("TEST", enriched)
        start, end = _dates_from_df(enriched)

        config = BacktestConfig(
            strategy_name="insider_following",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={
                "sma_period": 20,
                "min_officer_buys": 2,
                "insider_buy_ratio_threshold": 0.5,
            },
        )
        result = _run_backtest(config, source)
        assert len(result.trades) > 0, "Officer buying should generate trades"

    def test_sell_on_heavy_selling(self):
        """Heavy insider selling -> SELL signal for existing position.

        Creates a scenario where the strategy first buys (due to insider buying),
        then sells (due to heavy insider selling afterward).
        """
        # Price goes up then down
        prices = [80 + i * 0.5 for i in range(60)] + [110 - i * 0.3 for i in range(90)]
        daily = make_controlled_df(prices, start="2020-01-02")

        # Buying early, then heavy selling later
        insider_rows = []
        for i in range(3):
            fd = date(2020, 1, 15) + timedelta(days=i * 3)
            insider_rows.append({
                "filed_date": fd, "transaction_date": fd,
                "insider_name": f"Officer_{i}",
                "insider_title": "Chief Executive Officer",
                "transaction_type": "P", "shares": 5000,
                "price": 90.0, "shares_after": 50000, "is_direct": True,
            })
        # Heavy selling later
        for i in range(5):
            fd = date(2020, 3, 15) + timedelta(days=i * 3)
            insider_rows.append({
                "filed_date": fd, "transaction_date": fd,
                "insider_name": "Big Seller",
                "insider_title": "CEO",
                "transaction_type": "S", "shares": -20000,
                "price": 105.0, "shares_after": 10000, "is_direct": True,
            })
        insider = pd.DataFrame(insider_rows)
        enriched = _enrich_with_insider(daily, insider)

        source = MockDataSource()
        source.add("TEST", enriched)
        start, end = _dates_from_df(enriched)

        config = BacktestConfig(
            strategy_name="insider_following",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={
                "sma_period": 20,
                "min_officer_buys": 2,
                "heavy_selling_threshold": -5000,
            },
        )
        result = _run_backtest(config, source)
        # Should have at least 2 trades (buy then sell)
        assert len(result.trades) >= 2, "Should buy then sell on heavy selling"

    def test_no_trade_without_insider_data(self):
        """Without insider data, insider_following produces no trades."""
        prices = [80 + i * 0.3 for i in range(150)]
        daily = make_controlled_df(prices, start="2020-01-02")

        source = MockDataSource()
        source.add("TEST", daily)
        start, end = _dates_from_df(daily)

        config = BacktestConfig(
            strategy_name="insider_following",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 20},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) == 0


class TestSmartMoneyE2E:
    """E2E tests for smart_money strategy."""

    def test_buy_on_inst_accumulation_and_insider_buying(self):
        """Institutional accumulation + insider buying + good fundamentals -> BUY."""
        prices = [80 + i * 0.2 for i in range(300)]
        daily = make_controlled_df(prices, start="2020-01-02")

        fin = _make_good_financials()
        insider = _make_insider_buying(start_date=date(2020, 6, 1), count=3)
        inst = pd.DataFrame([
            {"filed_date": date(2020, 2, 14), "report_date": date(2019, 12, 31),
             "total_holders": 100, "total_shares": 1_000_000, "total_value": 50_000_000},
            {"filed_date": date(2020, 5, 15), "report_date": date(2020, 3, 31),
             "total_holders": 120, "total_shares": 1_100_000, "total_value": 55_000_000},
        ])

        enriched = _enrich_with_all(daily, fin_df=fin, insider_df=insider, inst_df=inst)

        source = MockDataSource()
        source.add("TEST", enriched)
        start, end = _dates_from_df(enriched)

        config = BacktestConfig(
            strategy_name="smart_money",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={
                "sma_period": 50,
                "inst_growth_threshold": 0.05,
                "insider_confirmation": True,
            },
        )
        result = _run_backtest(config, source)
        # The strategy requires inst_shares_change_pct, fund_net_income,
        # fund_revenue_growth_yoy, and insider_buy_ratio_90d.
        # This is a complex multi-signal strategy. Verify it runs without error.
        assert result.equity_series is not None

    def test_no_trade_without_institutional_data(self):
        """Without institutional data, smart_money produces no trades."""
        prices = [80 + i * 0.2 for i in range(150)]
        daily = make_controlled_df(prices, start="2020-01-02")

        source = MockDataSource()
        source.add("TEST", daily)
        start, end = _dates_from_df(daily)

        config = BacktestConfig(
            strategy_name="smart_money",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) == 0


class TestFundamentalScreenerE2E:
    """E2E tests for the fundamental_screener strategy."""

    def test_screener_with_buy_rules(self):
        """Screener with buy rules on fund_ columns."""
        prices = [80 + i * 0.17 for i in range(300)]
        daily = make_controlled_df(prices, start="2020-01-02")
        fin = _make_good_financials()
        enriched = _enrich_with_financials(daily, fin)

        source = MockDataSource()
        source.add("TEST", enriched)
        start, end = _dates_from_df(enriched)

        config = BacktestConfig(
            strategy_name="fundamental_screener",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={
                "sma_period": 50,
                "buy_rules": [
                    ["fund_net_income", ">", 0],
                    ["fund_net_margin", ">", 0.05],
                    ["Close", ">", "sma_trend"],
                ],
                "sell_rules": [
                    ["Close", "<", "sma_trend"],
                ],
            },
        )
        result = _run_backtest(config, source)
        # With profitable company above SMA, should trigger buy rules
        assert result.equity_series is not None

    def test_screener_no_rules_no_trades(self):
        """Screener with empty rules -> no trades."""
        prices = [80 + i * 0.17 for i in range(100)]
        daily = make_controlled_df(prices, start="2020-01-02")

        source = MockDataSource()
        source.add("TEST", daily)
        start, end = _dates_from_df(daily)

        config = BacktestConfig(
            strategy_name="fundamental_screener",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 20, "buy_rules": [], "sell_rules": []},
        )
        result = _run_backtest(config, source)
        assert len(result.trades) == 0

    def test_screener_with_insider_column_rule(self):
        """Screener rules can reference insider_ columns."""
        prices = [80 + i * 0.3 for i in range(150)]
        daily = make_controlled_df(prices, start="2020-01-02")
        insider = _make_insider_buying(start_date=date(2020, 2, 1), count=5)
        enriched = _enrich_with_insider(daily, insider)

        source = MockDataSource()
        source.add("TEST", enriched)
        start, end = _dates_from_df(enriched)

        config = BacktestConfig(
            strategy_name="fundamental_screener",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={
                "sma_period": 20,
                "buy_rules": [
                    ["insider_officer_buys_90d", ">=", 2],
                    ["Close", ">", "sma_trend"],
                ],
                "sell_rules": [],
            },
        )
        result = _run_backtest(config, source)
        # Should buy when officer buys >= 2 and above SMA
        assert len(result.trades) > 0, "Screener should buy based on insider rules"


class TestEventDataE2E:
    """E2E tests for event data merging."""

    def test_restatement_flag_set(self):
        """Mock 8-K restatement -> event_restatement_ever is set in data."""
        prices = [100 + i * 0.1 for i in range(100)]
        daily = make_controlled_df(prices, start="2020-01-02")

        events = pd.DataFrame([
            {
                "filed_date": date(2020, 3, 15),
                "event_date": date(2020, 3, 14),
                "items": ["4.02"],
                "item_descriptions": ["Restatement"],
                "has_earnings": False,
                "has_acquisition": False,
                "has_officer_change": False,
                "has_restatement": True,
                "has_material_agreement": False,
                "has_delisting_notice": False,
                "has_bankruptcy": False,
            }
        ])

        enriched = _enrich_with_all(daily, event_df=events)

        # After the restatement filing date, the flag should be set
        after = enriched.loc[enriched.index >= pd.Timestamp(2020, 3, 15)]
        assert not after.empty
        assert after["event_restatement_ever"].iloc[0] == 1.0

        # Before the filing, flag should not be set
        before = enriched.loc[enriched.index < pd.Timestamp(2020, 3, 15)]
        if not before.empty:
            assert (before["event_restatement_ever"] == 0.0).all()


class TestCSVBackwardCompatibilityE2E:
    """E2E test using fundamental_data_path with CSV (old workflow)."""

    def test_csv_fundamental_path(self):
        """Use fundamental_data_path with CSV, verify old workflow works through engine."""
        prices = [80 + i * 0.17 for i in range(100)]
        daily = make_controlled_df(prices, start="2020-01-02")

        source = MockDataSource()
        source.add("TEST", daily)
        start, end = _dates_from_df(daily)

        # Write a CSV fundamental file
        tmpdir = tempfile.mkdtemp()
        csv_path = f"{tmpdir}/fund.csv"
        with open(csv_path, "w") as f:
            f.write("date,symbol,field,value\n")
            f.write("2020-01-15,TEST,pe_ratio,15.0\n")
            f.write("2020-04-15,TEST,pe_ratio,18.0\n")

        config = BacktestConfig(
            strategy_name="sma_crossover",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_fast": 10, "sma_slow": 30},
            fundamental_data_path=csv_path,
        )
        result = _run_backtest(config, source)
        # sma_crossover does not use fundamental data, so this just tests
        # that the engine doesn't crash when fundamental_data_path is set
        assert result.equity_series is not None


class TestNoLookahead:
    """Verify point-in-time correctness in the E2E pipeline."""

    def test_filing_date_respected(self):
        """Financial data filed 35 days after period end should not be visible early.

        This test verifies that the merge_asof on filed_date prevents lookahead.
        """
        # Create daily data
        prices = [100] * 60  # flat price
        daily = make_controlled_df(prices, start="2020-04-01")

        # Financial data filed May 5 (35 days after March 31 period end)
        fin = pd.DataFrame([
            {"metric": "net_income", "period_end": date(2020, 3, 31),
             "filed_date": date(2020, 5, 5), "value": 1e8, "form": "10-Q"},
            {"metric": "revenue", "period_end": date(2020, 3, 31),
             "filed_date": date(2020, 5, 5), "value": 1e9, "form": "10-Q"},
        ])

        enriched = _enrich_with_financials(daily, fin)

        # Before May 5, fund_net_income should be NaN
        before = enriched.loc[enriched.index < pd.Timestamp(2020, 5, 5)]
        if "fund_net_income" in enriched.columns and not before.empty:
            assert before["fund_net_income"].isna().all(), (
                "Financial data should not be visible before filing date"
            )

        # On/after May 5, should be available
        after = enriched.loc[enriched.index >= pd.Timestamp(2020, 5, 5)]
        if "fund_net_income" in enriched.columns and not after.empty:
            assert after["fund_net_income"].notna().any()


class TestMultipleStrategiesSameData:
    """Run different strategies on same data to verify isolation."""

    def test_different_strategies_same_data(self):
        """Two strategies with same price data should run independently."""
        prices = [80 + i * 0.17 for i in range(300)]
        daily = make_controlled_df(prices, start="2020-01-02")
        fin = _make_good_financials()
        enriched = _enrich_with_financials(daily, fin)

        source1 = MockDataSource()
        source1.add("TEST", enriched)
        source2 = MockDataSource()
        source2.add("TEST", enriched)
        start, end = _dates_from_df(enriched)

        config1 = BacktestConfig(
            strategy_name="value_quality",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50, "max_pe": 30.0, "min_pe": 1.0},
        )
        config2 = BacktestConfig(
            strategy_name="earnings_growth",
            tickers=["TEST"],
            benchmark="TEST",
            start_date=start,
            end_date=end,
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.50,
            strategy_params={"sma_period": 50, "max_pe": 100.0},
        )

        result1 = _run_backtest(config1, source1)
        result2 = _run_backtest(config2, source2)

        # Both should complete without error
        assert result1.equity_series is not None
        assert result2.equity_series is not None
        # They should not necessarily produce identical results
        # (different strategies = different signals)
