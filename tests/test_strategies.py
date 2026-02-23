"""Tests for strategy framework and SMA crossover."""

import math
import pandas as pd
import pytest
from datetime import date

from backtester.strategies.registry import get_strategy, list_strategies
from backtester.strategies.indicators import (
    sma, rsi, macd, bollinger, stochastic, adx, obv,
    ema, atr, keltner, donchian, williams_r, cci, mfi, roc, psar, ichimoku, vwap,
)
from backtester.portfolio.portfolio import PortfolioState
from backtester.portfolio.position import Position
from backtester.types import SignalAction
from tests.conftest import make_price_df


class TestIndicators:
    def test_sma(self):
        prices = pd.Series([1, 2, 3, 4, 5], dtype=float)
        result = sma(prices, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 2.0  # (1+2+3)/3
        assert result.iloc[3] == 3.0
        assert result.iloc[4] == 4.0

    def test_rsi_range(self):
        df = make_price_df(days=100)
        result = rsi(df["Close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_ema(self):
        prices = pd.Series([1, 2, 3, 4, 5], dtype=float)
        result = ema(prices, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert not pd.isna(result.iloc[2])
        assert len(result) == 5

    def test_atr(self):
        df = make_price_df(days=50)
        result = atr(df, 14)
        assert len(result) == 50
        assert result.iloc[:13].isna().all()
        valid = result.dropna()
        assert (valid > 0).all()

    def test_keltner(self):
        df = make_price_df(days=50)
        upper, middle, lower = keltner(df, period=20, atr_period=14)
        valid_idx = upper.dropna().index
        assert len(valid_idx) > 0
        assert (upper.loc[valid_idx] > middle.loc[valid_idx]).all()
        assert (lower.loc[valid_idx] < middle.loc[valid_idx]).all()

    def test_donchian(self):
        df = make_price_df(days=50)
        upper, middle, lower = donchian(df, period=20)
        valid_idx = upper.dropna().index
        assert len(valid_idx) > 0
        assert (upper.loc[valid_idx] >= lower.loc[valid_idx]).all()

    def test_williams_r_range(self):
        df = make_price_df(days=50)
        result = williams_r(df, 14)
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid >= -100).all()
        assert (valid <= 0).all()

    def test_cci(self):
        df = make_price_df(days=50)
        result = cci(df, 20)
        assert len(result) == 50
        assert result.iloc[:19].isna().all()
        assert not result.dropna().empty

    def test_mfi_range(self):
        df = make_price_df(days=50)
        result = mfi(df, 14)
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_roc(self):
        prices = pd.Series([100, 110, 120, 130, 140], dtype=float)
        result = roc(prices, 2)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # roc[2] = (120-100)/100 * 100 = 20.0
        assert result.iloc[2] == pytest.approx(20.0)

    def test_psar(self):
        df = make_price_df(days=50)
        result = psar(df)
        assert len(result) == 50
        # SAR should be finite where computed
        valid = result.dropna()
        assert len(valid) > 0
        assert all(pd.notna(v) for v in valid)

    def test_psar_short_data(self):
        df = make_price_df(days=1)
        result = psar(df)
        assert len(result) == 1
        assert pd.isna(result.iloc[0])

    def test_ichimoku(self):
        df = make_price_df(days=100)
        t_sen, k_sen, s_a, s_b, chikou = ichimoku(df)
        assert len(t_sen) == 100
        # tenkan warmup = 9 periods
        assert t_sen.iloc[:8].isna().all()
        assert not t_sen.iloc[8:].isna().all()

    def test_vwap(self):
        df = make_price_df(days=50)
        result = vwap(df, period=20)
        assert len(result) == 50
        assert result.iloc[:19].isna().all()
        valid = result.dropna()
        assert (valid > 0).all()


class TestRegistry:
    def test_sma_crossover_registered(self):
        strategies = list_strategies()
        assert "sma_crossover" in strategies

    def test_get_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("nonexistent")

    def test_instantiate_strategy(self):
        s = get_strategy("sma_crossover")
        assert s is not None


class TestSmaCrossover:
    def test_configure(self):
        s = get_strategy("sma_crossover")
        s.configure({"sma_fast": 10, "sma_slow": 30})
        assert s.sma_fast == 10
        assert s.sma_slow == 30

    def test_configure_invalid(self):
        s = get_strategy("sma_crossover")
        with pytest.raises(ValueError):
            s.configure({"sma_fast": 200, "sma_slow": 100})

    def test_compute_indicators(self):
        s = get_strategy("sma_crossover")
        s.configure({"sma_fast": 5, "sma_slow": 10})
        df = make_price_df(days=50)
        result = s.compute_indicators(df)
        assert "sma_fast" in result.columns
        assert "sma_slow" in result.columns

    def test_buy_signal(self):
        s = get_strategy("sma_crossover")
        row = pd.Series({"Close": 100, "sma_fast": 110, "sma_slow": 105})
        state = PortfolioState(cash=100_000, total_equity=100_000,
                               num_positions=0, position_symbols=frozenset())
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.BUY

    def test_sell_signal(self):
        s = get_strategy("sma_crossover")
        pos = Position(symbol="TEST")
        pos.add_lot(100, 100.0, date(2020, 1, 2))

        row = pd.Series({"Close": 100, "sma_fast": 95, "sma_slow": 105})
        state = PortfolioState(cash=50_000, total_equity=60_000,
                               num_positions=1, position_symbols=frozenset({"TEST"}))
        signal = s.generate_signals("TEST", row, pos, state)
        assert signal == SignalAction.SELL

    def test_hold_during_warmup(self):
        s = get_strategy("sma_crossover")
        row = pd.Series({"Close": 100, "sma_fast": float("nan"), "sma_slow": float("nan")})
        state = PortfolioState(cash=100_000, total_equity=100_000,
                               num_positions=0, position_symbols=frozenset())
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.HOLD


# ---------------------------------------------------------------------------
# MACD indicator tests
# ---------------------------------------------------------------------------

class TestMacd:
    def test_returns_three_series(self):
        prices = pd.Series(range(1, 101), dtype=float)
        result = macd(prices)
        assert isinstance(result, tuple) and len(result) == 3
        for s in result:
            assert isinstance(s, pd.Series)

    def test_warmup_produces_nan(self):
        prices = pd.Series(range(1, 101), dtype=float)
        macd_line, signal_line, histogram = macd(prices, fast=12, slow=26, signal=9)
        # First 25 values of macd_line should be NaN (slow EMA needs 26 periods)
        assert macd_line.iloc[:25].isna().all()
        # Signal line needs additional warmup on top of macd_line
        assert signal_line.iloc[:33].isna().all()

    def test_directional_correctness(self):
        # Steadily rising prices → MACD line should be positive once warmed up
        prices = pd.Series([50.0 + i * 0.5 for i in range(100)])
        macd_line, _, _ = macd(prices, fast=12, slow=26, signal=9)
        valid = macd_line.dropna()
        assert (valid > 0).all()


# ---------------------------------------------------------------------------
# Rule-based strategy tests
# ---------------------------------------------------------------------------

def _make_state(num_positions=0, symbols=frozenset()):
    return PortfolioState(
        cash=100_000, total_equity=100_000,
        num_positions=num_positions, position_symbols=symbols,
    )


class TestRuleBasedStrategy:
    def test_configure_indicators(self):
        s = get_strategy("rule_based")
        s.configure({
            "indicators": {"rsi": {"period": 14}},
            "buy_when": [["rsi", ">", 60]],
            "sell_when": [["rsi", "<", 40]],
        })
        # Should not raise

    def test_configure_invalid_fn(self):
        s = get_strategy("rule_based")
        with pytest.raises(ValueError, match="Unknown indicator"):
            s.configure({
                "indicators": {"x": {"fn": "bogus", "period": 5}},
                "buy_when": [],
                "sell_when": [],
            })

    def test_configure_invalid_operator(self):
        s = get_strategy("rule_based")
        with pytest.raises(ValueError, match="unknown operator"):
            s.configure({
                "indicators": {},
                "buy_when": [["Close", "!!", 100]],
                "sell_when": [],
            })

    def test_buy_signal_all_rules_met(self):
        s = get_strategy("rule_based")
        s.configure({
            "indicators": {"rsi": {"period": 14}},
            "buy_when": [["rsi", ">", 60], ["Volume", ">=", 500000]],
            "sell_when": [["rsi", "<", 40]],
        })
        row = pd.Series({"Close": 100, "rsi": 65.0, "Volume": 1_000_000})
        signal = s.generate_signals("TEST", row, None, _make_state())
        assert signal == SignalAction.BUY

    def test_buy_blocked_one_rule_fails(self):
        s = get_strategy("rule_based")
        s.configure({
            "indicators": {"rsi": {"period": 14}},
            "buy_when": [["rsi", ">", 60], ["Volume", ">=", 500000]],
            "sell_when": [],
        })
        # RSI met, but volume too low
        row = pd.Series({"Close": 100, "rsi": 65.0, "Volume": 100_000})
        signal = s.generate_signals("TEST", row, None, _make_state())
        assert signal == SignalAction.HOLD

    def test_sell_signal(self):
        s = get_strategy("rule_based")
        s.configure({
            "indicators": {"rsi": {"period": 14}},
            "buy_when": [["rsi", ">", 60]],
            "sell_when": [["rsi", "<", 40]],
        })
        pos = Position(symbol="TEST")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        row = pd.Series({"Close": 90, "rsi": 30.0})
        state = _make_state(num_positions=1, symbols=frozenset({"TEST"}))
        signal = s.generate_signals("TEST", row, pos, state)
        assert signal == SignalAction.SELL

    def test_warmup_hold(self):
        s = get_strategy("rule_based")
        s.configure({
            "indicators": {"rsi": {"period": 14}},
            "buy_when": [["rsi", ">", 60]],
            "sell_when": [],
        })
        row = pd.Series({"Close": 100, "rsi": float("nan")})
        signal = s.generate_signals("TEST", row, None, _make_state())
        assert signal == SignalAction.HOLD

    def test_benchmark_column_access(self):
        s = get_strategy("rule_based")
        s.configure({
            "indicators": {},
            "benchmark_indicators": {"sma_200": {"fn": "sma", "period": 200}},
            "buy_when": [["Close", ">", 100], ["bm_Close", ">", "bm_sma_200"]],
            "sell_when": [],
        })
        row = pd.Series({"Close": 110.0})
        bm_row = pd.Series({"Close": 300.0, "sma_200": 290.0})
        signal = s.generate_signals("TEST", row, None, _make_state(), benchmark_row=bm_row)
        assert signal == SignalAction.BUY

    def test_column_vs_column_comparison(self):
        s = get_strategy("rule_based")
        s.configure({
            "indicators": {
                "sma_fast": {"fn": "sma", "period": 50},
                "sma_slow": {"fn": "sma", "period": 200},
            },
            "buy_when": [["sma_fast", ">", "sma_slow"]],
            "sell_when": [["sma_fast", "<", "sma_slow"]],
        })
        # Buy: fast > slow
        row = pd.Series({"Close": 100, "sma_fast": 110.0, "sma_slow": 105.0})
        signal = s.generate_signals("TEST", row, None, _make_state())
        assert signal == SignalAction.BUY

        # Sell: fast < slow (with position)
        pos = Position(symbol="TEST")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        row2 = pd.Series({"Close": 100, "sma_fast": 95.0, "sma_slow": 105.0})
        state = _make_state(num_positions=1, symbols=frozenset({"TEST"}))
        signal2 = s.generate_signals("TEST", row2, pos, state)
        assert signal2 == SignalAction.SELL

    def test_macd_indicator_expands(self):
        s = get_strategy("rule_based")
        s.configure({
            "indicators": {"macd": {"fn": "macd", "fast": 12, "slow": 26, "signal": 9}},
            "buy_when": [["macd_line", ">", 0]],
            "sell_when": [],
        })
        df = make_price_df(days=100)
        result = s.compute_indicators(df)
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

    def test_bollinger_indicator_expands(self):
        s = get_strategy("rule_based")
        s.configure({
            "indicators": {"bb": {"fn": "bollinger", "period": 20}},
            "buy_when": [["Close", "<", "bb_lower"]],
            "sell_when": [],
        })
        df = make_price_df(days=100)
        result = s.compute_indicators(df)
        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns

    def test_stochastic_indicator_expands(self):
        s = get_strategy("rule_based")
        s.configure({
            "indicators": {"stoch": {"fn": "stochastic", "k_period": 14}},
            "buy_when": [["stoch_k", "<", 20]],
            "sell_when": [],
        })
        df = make_price_df(days=100)
        result = s.compute_indicators(df)
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns


# ---------------------------------------------------------------------------
# Bollinger Bands tests
# ---------------------------------------------------------------------------

class TestBollinger:
    def test_returns_three_series(self):
        prices = pd.Series(range(1, 101), dtype=float)
        result = bollinger(prices, period=20)
        assert isinstance(result, tuple) and len(result) == 3
        for s in result:
            assert isinstance(s, pd.Series)

    def test_warmup_produces_nan(self):
        prices = pd.Series(range(1, 101), dtype=float)
        upper, middle, lower = bollinger(prices, period=20)
        assert upper.iloc[:19].isna().all()
        assert middle.iloc[:19].isna().all()
        assert lower.iloc[:19].isna().all()

    def test_middle_equals_sma(self):
        prices = pd.Series(range(1, 101), dtype=float)
        _, middle, _ = bollinger(prices, period=20)
        expected_sma = sma(prices, 20)
        pd.testing.assert_series_equal(middle, expected_sma)


# ---------------------------------------------------------------------------
# Stochastic Oscillator tests
# ---------------------------------------------------------------------------

class TestStochastic:
    def test_returns_two_series(self):
        df = make_price_df(days=100)
        result = stochastic(df, k_period=14, d_period=3)
        assert isinstance(result, tuple) and len(result) == 2
        for s in result:
            assert isinstance(s, pd.Series)

    def test_values_in_range(self):
        df = make_price_df(days=100)
        k, d = stochastic(df, k_period=14, d_period=3)
        valid_k = k.dropna()
        valid_d = d.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    def test_warmup_produces_nan(self):
        df = make_price_df(days=100)
        k, d = stochastic(df, k_period=14, d_period=3)
        assert k.iloc[:13].isna().all()
        assert d.iloc[:15].isna().all()


# ---------------------------------------------------------------------------
# ADX tests
# ---------------------------------------------------------------------------

class TestADX:
    def test_returns_series(self):
        df = make_price_df(days=100)
        result = adx(df, period=14)
        assert isinstance(result, pd.Series)

    def test_values_in_range(self):
        df = make_price_df(days=100)
        result = adx(df, period=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_trending_data_high_adx(self):
        # Strong uptrend should produce a meaningful ADX
        dates = pd.bdate_range("2020-01-02", periods=100, freq="B")
        prices = [100.0 + i * 2.0 for i in range(100)]
        df = pd.DataFrame({
            "Open": [p - 0.5 for p in prices],
            "High": [p + 1.0 for p in prices],
            "Low": [p - 1.0 for p in prices],
            "Close": prices,
        }, index=dates)
        result = adx(df, period=14)
        # Last value should indicate strong trend
        assert result.iloc[-1] > 25


# ---------------------------------------------------------------------------
# OBV tests
# ---------------------------------------------------------------------------

class TestOBV:
    def test_returns_series(self):
        df = make_price_df(days=50)
        result = obv(df)
        assert isinstance(result, pd.Series)

    def test_rising_prices_positive_obv(self):
        dates = pd.bdate_range("2020-01-02", periods=20, freq="B")
        prices = [100.0 + i for i in range(20)]
        df = pd.DataFrame({
            "Close": prices,
            "Volume": [1_000_000] * 20,
        }, index=dates)
        result = obv(df)
        # All prices rising → cumulative OBV should be positive at end
        assert result.iloc[-1] > 0

    def test_no_params_required(self):
        df = make_price_df(days=30)
        # Should work with no extra params
        result = obv(df)
        assert len(result) == 30
