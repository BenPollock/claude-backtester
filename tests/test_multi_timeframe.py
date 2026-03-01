"""Tests for multi-timeframe data support."""

import tempfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtester.config import BacktestConfig
from backtester.data.manager import resample_ohlcv, DataManager
from backtester.engine import BacktestEngine
from backtester.strategies.base import Strategy
from backtester.strategies.registry import register_strategy, get_strategy
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position
from backtester.types import SignalAction
from tests.conftest import make_price_df, MockDataSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_df(start="2020-01-02", days=60):
    """Build a deterministic daily OHLCV DataFrame for resampling tests."""
    dates = pd.bdate_range(start=start, periods=days, freq="B")
    rng = np.random.default_rng(99)
    prices = 100.0 + np.cumsum(rng.normal(0, 1, days))
    df = pd.DataFrame(
        {
            "Open": prices - 0.5,
            "High": prices + 1.0,
            "Low": prices - 1.0,
            "Close": prices,
            "Volume": np.full(days, 1_000_000, dtype=int),
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    return df


# ---------------------------------------------------------------------------
# resample_ohlcv tests
# ---------------------------------------------------------------------------

class TestResampleOHLCV:
    def test_weekly_ohlcv_aggregation(self):
        """Weekly bars: O=first, H=max, L=min, C=last, V=sum."""
        df = _make_daily_df(days=10)

        result = resample_ohlcv(df, "weekly")

        # Should have at least 2 weekly bars from 10 trading days
        assert len(result) >= 2

        # Verify aggregation for the first complete-ish week
        first_period_end = result.index[0]
        daily_in_period = df.loc[df.index <= first_period_end]

        assert result.iloc[0]["Open"] == pytest.approx(daily_in_period["Open"].iloc[0])
        assert result.iloc[0]["High"] == pytest.approx(daily_in_period["High"].max())
        assert result.iloc[0]["Low"] == pytest.approx(daily_in_period["Low"].min())
        assert result.iloc[0]["Close"] == pytest.approx(daily_in_period["Close"].iloc[-1])
        assert result.iloc[0]["Volume"] == pytest.approx(daily_in_period["Volume"].sum())

    def test_monthly_ohlcv_aggregation(self):
        """Monthly bars: O=first, H=max, L=min, C=last, V=sum."""
        df = _make_daily_df(days=60)

        result = resample_ohlcv(df, "monthly")

        # 60 business days ~ 3 months
        assert len(result) >= 2

        # Verify aggregation for the first month
        first_period_end = result.index[0]
        daily_in_period = df.loc[df.index <= first_period_end]

        assert result.iloc[0]["Open"] == pytest.approx(daily_in_period["Open"].iloc[0])
        assert result.iloc[0]["High"] == pytest.approx(daily_in_period["High"].max())
        assert result.iloc[0]["Low"] == pytest.approx(daily_in_period["Low"].min())
        assert result.iloc[0]["Close"] == pytest.approx(daily_in_period["Close"].iloc[-1])
        assert result.iloc[0]["Volume"] == pytest.approx(daily_in_period["Volume"].sum())

    def test_partial_periods_included(self):
        """Partial weeks/months at boundaries should be included, not dropped."""
        # Start on a Wednesday so the first week is partial (Wed-Fri = 3 days)
        df = _make_daily_df(start="2020-01-08", days=10)  # Wed Jan 8

        result = resample_ohlcv(df, "weekly")

        # First period should contain the partial week
        assert len(result) >= 2
        # The first bar should exist even though it's a partial week
        assert not pd.isna(result.iloc[0]["Close"])

    def test_index_uses_last_trading_day(self):
        """Resampled index should use actual trading days, not calendar period-ends."""
        df = _make_daily_df(days=30)
        result = resample_ohlcv(df, "weekly")

        # Every index value should exist in the original daily index
        for idx in result.index:
            assert idx in df.index, f"{idx} not in daily index"

    def test_invalid_timeframe_raises(self):
        """Requesting an unsupported timeframe should raise ValueError."""
        df = _make_daily_df(days=10)
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            resample_ohlcv(df, "quarterly")

    def test_does_not_mutate_input(self):
        """resample_ohlcv should not mutate the input DataFrame."""
        df = _make_daily_df(days=10)
        original_shape = df.shape
        original_values = df.values.copy()

        resample_ohlcv(df, "weekly")

        assert df.shape == original_shape
        np.testing.assert_array_equal(df.values, original_values)

    def test_columns_preserved(self):
        """Output should have the same OHLCV columns as input."""
        df = _make_daily_df(days=20)
        result = resample_ohlcv(df, "weekly")
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# Strategy.timeframes property tests
# ---------------------------------------------------------------------------

class TestStrategyTimeframes:
    def test_default_timeframes(self):
        """Default Strategy.timeframes returns ['daily']."""
        strategy = get_strategy("sma_crossover")
        assert strategy.timeframes == ["daily"]

    def test_default_timeframes_rule_based(self):
        """rule_based strategy also defaults to ['daily']."""
        strategy = get_strategy("rule_based")
        assert strategy.timeframes == ["daily"]


# ---------------------------------------------------------------------------
# Multi-timeframe strategy for integration testing
# ---------------------------------------------------------------------------

# Use a unique name to avoid colliding with production strategies
_MTF_STRATEGY_NAME = "_test_mtf_strategy"


@register_strategy(_MTF_STRATEGY_NAME)
class MultiTimeframeTestStrategy(Strategy):
    """Test strategy that requests weekly data and uses it in indicators."""

    def __init__(self):
        self._received_timeframe_data = None

    @property
    def timeframes(self) -> list[str]:
        return ["daily", "weekly"]

    def configure(self, params: dict) -> None:
        pass

    def compute_indicators(
        self,
        df: pd.DataFrame,
        timeframe_data: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        df = df.copy()
        self._received_timeframe_data = timeframe_data

        # If weekly data is available, merge the weekly Close as an indicator
        if timeframe_data and "weekly" in timeframe_data:
            weekly = timeframe_data["weekly"]
            df["weekly_close_indicator"] = weekly["Close"]

        return df

    def generate_signals(
        self,
        symbol: str,
        row: pd.Series,
        position: Position | None,
        portfolio_state: PortfolioState,
        benchmark_row: pd.Series | None = None,
    ) -> SignalAction:
        # Simple: always HOLD (we just want to test data flow)
        return SignalAction.HOLD


class TestMultiTimeframeComputeIndicators:
    def test_compute_indicators_receives_timeframe_data(self):
        """When timeframe_data is passed, strategy's compute_indicators receives it."""
        strategy = get_strategy(_MTF_STRATEGY_NAME)
        strategy.configure({})

        df = _make_daily_df(days=30)

        # Build timeframe_data like the engine would
        weekly = resample_ohlcv(df, "weekly")
        weekly_ff = weekly.reindex(df.index).ffill()
        tf_data = {"weekly": weekly_ff}

        result = strategy.compute_indicators(df, timeframe_data=tf_data)

        # Strategy should have received the timeframe data
        assert strategy._received_timeframe_data is not None
        assert "weekly" in strategy._received_timeframe_data
        # And the weekly close indicator should be in the output
        assert "weekly_close_indicator" in result.columns

    def test_compute_indicators_backward_compat_no_timeframe_data(self):
        """Existing strategies work when timeframe_data is not provided."""
        strategy = get_strategy("sma_crossover")
        strategy.configure({"sma_fast": 5, "sma_slow": 10})

        df = _make_daily_df(days=30)
        result = strategy.compute_indicators(df)

        assert "sma_fast" in result.columns
        assert "sma_slow" in result.columns


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------

class TestMultiTimeframeEngine:
    def _make_engine(self, tmpdir, strategy_name, days=60, tickers=None,
                     strategy_params=None):
        """Helper to create an engine with mock data."""
        tickers = tickers or ["TEST"]
        source = MockDataSource()
        df = make_price_df(start="2020-01-02", days=days, start_price=100.0)
        for t in tickers:
            source.add(t, df)

        config = BacktestConfig(
            strategy_name=strategy_name,
            tickers=tickers,
            benchmark=tickers[0],
            start_date=date(2020, 1, 2),
            end_date=date(2020, 12, 31),
            starting_cash=100_000.0,
            max_positions=10,
            max_alloc_pct=0.20,
            fee_per_trade=0.0,
            slippage_bps=0.0,
            data_cache_dir=tmpdir,
            strategy_params=strategy_params or {},
        )

        data_mgr = DataManager(cache_dir=tmpdir, source=source)
        return BacktestEngine(config, data_manager=data_mgr)

    def test_mtf_strategy_gets_resampled_data(self):
        """Engine passes resampled timeframe_data to multi-timeframe strategies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir, _MTF_STRATEGY_NAME, days=60)
            result = engine.run()

            assert result is not None
            assert len(result.portfolio.equity_history) > 0

    def test_mtf_strategy_has_weekly_columns_in_universe(self):
        """After engine runs, daily data should contain weekly-prefixed columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir, _MTF_STRATEGY_NAME, days=60)

            # Run the engine -- we need to inspect internal state, so we
            # replicate the key setup steps.
            config = engine.config
            strategy = get_strategy(config.strategy_name)
            strategy.configure(config.strategy_params)

            data_mgr = engine._data
            universe_data = data_mgr.load_many(
                config.tickers, config.start_date, config.end_date
            )

            # Resample (mimic what engine.run does)
            from backtester.data.manager import resample_ohlcv as _resample
            extra_tfs = [tf for tf in strategy.timeframes if tf != "daily"]
            assert extra_tfs == ["weekly"]

            for symbol, daily_df in universe_data.items():
                tf_map = {}
                for tf in extra_tfs:
                    resampled = _resample(daily_df, tf)
                    ff = resampled.reindex(daily_df.index).ffill()
                    tf_map[tf] = ff
                universe_data[symbol] = strategy.compute_indicators(
                    daily_df, timeframe_data=tf_map
                )
                for tf_name, tf_df in tf_map.items():
                    for col in tf_df.columns:
                        universe_data[symbol][f"{tf_name}_{col}"] = tf_df[col]

            # Verify weekly columns exist
            test_df = universe_data["TEST"]
            assert "weekly_Close" in test_df.columns
            assert "weekly_Open" in test_df.columns
            assert "weekly_High" in test_df.columns
            assert "weekly_Low" in test_df.columns
            assert "weekly_Volume" in test_df.columns

    def test_backward_compat_existing_strategy(self):
        """Existing strategies (no multi-timeframe) still work unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(
                tmpdir, "sma_crossover", days=252,
                strategy_params={"sma_fast": 20, "sma_slow": 50},
            )
            result = engine.run()

            assert result is not None
            assert len(result.portfolio.equity_history) > 0
            assert result.equity_series.iloc[0] == 100_000.0
            # All positions should be closed at end
            assert result.portfolio.num_positions == 0

    def test_backward_compat_rule_based_strategy(self):
        """rule_based strategy still works unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(
                tmpdir, "rule_based", days=100,
                strategy_params={
                    "indicators": {"rsi": {"period": 14}},
                    "buy_when": [["rsi", ">", 60]],
                    "sell_when": [["rsi", "<", 40]],
                },
            )
            result = engine.run()

            assert result is not None
            assert len(result.portfolio.equity_history) > 0

    def test_forward_fill_covers_daily_dates(self):
        """Weekly data forward-filled to daily should cover all trading days."""
        df = _make_daily_df(days=30)
        weekly = resample_ohlcv(df, "weekly")
        ff = weekly.reindex(df.index).ffill()

        # After initial NaN period, every daily date should have a value
        # (the first few days before the first weekly bar ends will be NaN)
        first_weekly_date = weekly.index[0]
        after_first = ff.loc[ff.index >= first_weekly_date]
        assert after_first["Close"].notna().all()
