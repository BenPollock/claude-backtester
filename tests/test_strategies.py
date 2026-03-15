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

    def test_rsi_known_values(self):
        """RSI with alternating gains and losses should approach 50."""
        # Create 30 data points alternating +2, -2 to get roughly equal gains/losses
        prices = [50.0]
        for i in range(29):
            if i % 2 == 0:
                prices.append(prices[-1] + 2.0)
            else:
                prices.append(prices[-1] - 2.0)
        series = pd.Series(prices, dtype=float)
        result = rsi(series, 14)
        # With equal magnitude gains and losses, RSI should be near 50
        last_valid = result.dropna().iloc[-1]
        assert last_valid == pytest.approx(50.0, abs=10.0)

    def test_rsi_all_gains(self):
        """Monotonically increasing prices: avg_loss=0 => RSI should be 100."""
        prices = pd.Series([100.0 + i * 1.0 for i in range(30)])
        result = rsi(prices, 14)
        # First 13 values are NaN (min_periods=14 warmup), rest should be 100
        assert result.iloc[:13].isna().all()
        valid = result.dropna()
        assert len(valid) > 0
        for v in valid:
            assert v == pytest.approx(100.0)

    def test_rsi_mostly_gains(self):
        """Mostly gains with one tiny loss should give RSI near 100."""
        # Put a small loss in the first 14 diffs so avg_loss is non-zero
        prices = [100.0 + i for i in range(30)]
        prices[5] = prices[4] - 0.01  # one tiny loss within the first 14 periods
        series = pd.Series(prices, dtype=float)
        result = rsi(series, 14)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.iloc[-1] > 90.0

    def test_rsi_all_losses(self):
        """Monotonically decreasing prices should give RSI near 0."""
        prices = pd.Series([200.0 - i * 1.0 for i in range(30)])
        result = rsi(prices, 14)
        valid = result.dropna()
        assert len(valid) > 0
        # All diffs are -1 => avg_gain approaches 0 => RSI approaches 0
        assert valid.iloc[-1] < 10.0

    def test_ema(self):
        prices = pd.Series([1, 2, 3, 4, 5], dtype=float)
        result = ema(prices, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # ewm(span=3, min_periods=3, adjust=False): alpha = 2/(3+1) = 0.5
        # pandas adjust=False starts from first value and applies recursively
        # but min_periods=3 means first two are NaN
        # At index 2, the internal state has processed values 1, 2, 3 with adjust=False:
        # EMA[0]=1.0, EMA[1]=0.5*2+0.5*1=1.5, EMA[2]=0.5*3+0.5*1.5=2.25
        assert result.iloc[2] == pytest.approx(2.25, abs=0.01)
        # EMA[3] = 0.5*4 + 0.5*2.25 = 3.125
        assert result.iloc[3] == pytest.approx(3.125, abs=0.01)
        # EMA[4] = 0.5*5 + 0.5*3.125 = 4.0625
        assert result.iloc[4] == pytest.approx(4.0625, abs=0.01)

    def test_atr(self):
        df = make_price_df(days=50)
        result = atr(df, 14)
        assert len(result) == 50
        assert result.iloc[:13].isna().all()
        valid = result.dropna()
        assert (valid > 0).all()

    def test_atr_known_values(self):
        """ATR with known OHLC data should match manual calculation."""
        # Simple case: constant range, no gaps
        dates = pd.bdate_range("2020-01-02", periods=16, freq="B")
        df = pd.DataFrame({
            "Open": [100.0] * 16,
            "High": [105.0] * 16,
            "Low": [95.0] * 16,
            "Close": [100.0] * 16,
        }, index=dates)
        result = atr(df, 14)
        # TR = max(H-L, |H-prevC|, |L-prevC|) = max(10, 5, 5) = 10
        # First row TR: H-L=10, H-NaN=NaN, L-NaN=NaN => TR[0] = 10
        # All subsequent TR = 10. ATR(14) at index 14 = mean of 14 TRs = 10.0
        assert result.iloc[14] == pytest.approx(10.0, abs=0.01)

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

    def test_psar_known_uptrend(self):
        """PSAR in a steady uptrend: SAR should be below price."""
        dates = pd.bdate_range("2020-01-02", periods=20, freq="B")
        prices = [100.0 + i * 2.0 for i in range(20)]
        df = pd.DataFrame({
            "Open": [p - 0.5 for p in prices],
            "High": [p + 1.0 for p in prices],
            "Low": [p - 1.0 for p in prices],
            "Close": prices,
        }, index=dates)
        result = psar(df)
        # In a steady uptrend, SAR should be below the Low for most bars
        valid = result.dropna()
        assert len(valid) > 0
        # After initial warmup, SAR should be below Close
        assert (valid.iloc[2:] < df["Close"].loc[valid.index[2:]]).all()

    def test_psar_short_data(self):
        df = make_price_df(days=1)
        result = psar(df)
        assert len(result) == 1
        assert pd.isna(result.iloc[0])

    def test_psar_two_bars(self):
        """PSAR with exactly 2 bars should produce values for both."""
        dates = pd.bdate_range("2020-01-02", periods=2, freq="B")
        df = pd.DataFrame({
            "Open": [100.0, 102.0],
            "High": [105.0, 107.0],
            "Low": [95.0, 97.0],
            "Close": [102.0, 105.0],
        }, index=dates)
        result = psar(df)
        assert len(result) == 2
        # First SAR value should be set (bullish: starts at Low[0])
        assert pd.notna(result.iloc[0])
        assert result.iloc[0] == pytest.approx(95.0)  # initial SAR = Low[0] in bull

    def test_ichimoku(self):
        df = make_price_df(days=100)
        t_sen, k_sen, s_a, s_b, chikou = ichimoku(df)
        assert len(t_sen) == 100
        # tenkan warmup = 9 periods
        assert t_sen.iloc[:8].isna().all()
        assert not t_sen.iloc[8:].isna().all()

    def test_ichimoku_known_values(self):
        """Verify Tenkan-sen = (9-period high + 9-period low) / 2."""
        dates = pd.bdate_range("2020-01-02", periods=20, freq="B")
        highs = [float(100 + i) for i in range(20)]
        lows = [float(90 + i) for i in range(20)]
        closes = [float(95 + i) for i in range(20)]
        df = pd.DataFrame({
            "Open": closes,
            "High": highs,
            "Low": lows,
            "Close": closes,
        }, index=dates)
        t_sen, k_sen, s_a, s_b, chikou = ichimoku(df)
        # At index 8 (9th bar): High max = 108, Low min = 90
        # Tenkan = (108 + 90) / 2 = 99.0
        assert t_sen.iloc[8] == pytest.approx(99.0)
        # At index 9: High max = 109, Low min = 91 => (109+91)/2 = 100.0
        assert t_sen.iloc[9] == pytest.approx(100.0)
        # Chikou span is Close shifted forward by -26 periods (i.e., index 0
        # gets the value from index 26). With only 20 bars, chikou at index 0
        # should be NaN since there is no bar at index+26.
        assert pd.isna(chikou.iloc[0])

    def test_vwap(self):
        df = make_price_df(days=50)
        result = vwap(df, period=20)
        assert len(result) == 50
        assert result.iloc[:19].isna().all()
        valid = result.dropna()
        assert (valid > 0).all()

    def test_vwap_known_values(self):
        """VWAP with constant volume reduces to average of typical price."""
        dates = pd.bdate_range("2020-01-02", periods=5, freq="B")
        df = pd.DataFrame({
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [102.0, 103.0, 104.0, 105.0, 106.0],
            "Low":  [98.0,  99.0, 100.0, 101.0, 102.0],
            "Close": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Volume": [1000] * 5,
        }, index=dates)
        result = vwap(df, period=3)
        # Typical price = (H+L+C)/3
        # TP[0]=(102+98+101)/3=100.333, TP[1]=(103+99+102)/3=101.333
        # TP[2]=(104+100+103)/3=102.333
        # VWAP[2] = sum(TP*Vol)/sum(Vol) for window [0:3]
        # = (100.333*1000 + 101.333*1000 + 102.333*1000) / 3000
        # = 304000/3000 = 101.333...
        expected = (100.333333 + 101.333333 + 102.333333) / 3
        assert result.iloc[2] == pytest.approx(expected, abs=0.01)


    def test_indicators_all_nan_input(self):
        """Indicators should handle all-NaN input gracefully (return NaN, not crash)."""
        nan_series = pd.Series([float("nan")] * 20)
        result = sma(nan_series, 5)
        assert result.isna().all()
        result = ema(nan_series, 5)
        assert result.isna().all()
        result = rsi(nan_series, 14)
        assert result.isna().all()


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
        # sma_fast at index 4 should equal mean of first 5 Close prices
        expected_fast = df["Close"].iloc[:5].mean()
        assert result["sma_fast"].iloc[4] == pytest.approx(expected_fast, rel=1e-6)
        # sma_slow at index 9 should equal mean of first 10 Close prices
        expected_slow = df["Close"].iloc[:10].mean()
        assert result["sma_slow"].iloc[9] == pytest.approx(expected_slow, rel=1e-6)
        # Original df should not be mutated
        assert "sma_fast" not in df.columns

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


# ---------------------------------------------------------------------------
# Edge case indicator tests
# ---------------------------------------------------------------------------

class TestIndicatorEdgeCases:
    def test_stochastic_flat_prices(self):
        """Stochastic on flat prices: high_max == low_min causes division by zero.
        Should produce NaN, not crash."""
        dates = pd.bdate_range("2020-01-02", periods=20, freq="B")
        df = pd.DataFrame({
            "Open": [100.0] * 20,
            "High": [100.0] * 20,
            "Low": [100.0] * 20,
            "Close": [100.0] * 20,
        }, index=dates)
        k, d = stochastic(df, k_period=14, d_period=3)
        # With flat prices, high_max - low_min = 0, so %K should be NaN
        valid_k = k.dropna()
        # Either all NaN or if pandas produces 0/0 = NaN, that's fine
        # The key assertion: no exception raised AND values are NaN (not inf)
        for val in k.iloc[13:]:  # after warmup
            assert pd.isna(val) or math.isfinite(val)

    def test_adx_sideways_no_directional_movement(self):
        """ADX with no directional movement (same high/low each day).
        Should produce low ADX values, not crash."""
        dates = pd.bdate_range("2020-01-02", periods=60, freq="B")
        df = pd.DataFrame({
            "Open": [100.0] * 60,
            "High": [101.0] * 60,
            "Low": [99.0] * 60,
            "Close": [100.0] * 60,
        }, index=dates)
        result = adx(df, period=14)
        valid = result.dropna()
        # With zero directional movement (no higher highs, no lower lows),
        # plus_dm and minus_dm are both 0, making DX = 0/0 = NaN.
        # ADX should be NaN or very low (near 0).
        for val in valid:
            assert pd.isna(val) or val < 5.0, (
                f"ADX should be near 0 with no directional movement, got {val}"
            )

    def test_williams_r_flat_prices(self):
        """Williams %R on flat prices: denominator is zero.
        Should produce NaN, not crash or inf."""
        dates = pd.bdate_range("2020-01-02", periods=20, freq="B")
        df = pd.DataFrame({
            "Open": [50.0] * 20,
            "High": [50.0] * 20,
            "Low": [50.0] * 20,
            "Close": [50.0] * 20,
        }, index=dates)
        result = williams_r(df, 14)
        # After warmup (14 periods), denominator = 0, result should be NaN
        for val in result.iloc[13:]:
            assert pd.isna(val), (
                f"Williams %R should be NaN with flat prices, got {val}"
            )

    def test_cci_zero_volatility(self):
        """CCI with all same price: mean absolute deviation is 0.
        Should produce NaN, not crash or inf."""
        dates = pd.bdate_range("2020-01-02", periods=30, freq="B")
        df = pd.DataFrame({
            "Open": [75.0] * 30,
            "High": [75.0] * 30,
            "Low": [75.0] * 30,
            "Close": [75.0] * 30,
        }, index=dates)
        result = cci(df, 20)
        # Typical price = 75 for all bars, SMA of TP = 75, TP - SMA = 0
        # MAD = 0, denominator = 0 * 0.015 = 0 -> NaN
        for val in result.iloc[19:]:
            assert pd.isna(val) or val == pytest.approx(0.0, abs=1e-10), (
                f"CCI should be NaN or 0 with zero volatility, got {val}"
            )

    def test_mfi_all_positive_movement(self):
        """MFI with monotonically increasing typical price.
        All money flow is positive -> MFI should be near 100 (or NaN from 0 neg_sum)."""
        dates = pd.bdate_range("2020-01-02", periods=30, freq="B")
        prices = [100.0 + i * 1.0 for i in range(30)]
        df = pd.DataFrame({
            "Open": prices,
            "High": [p + 1.0 for p in prices],
            "Low": [p - 0.5 for p in prices],
            "Close": [p + 0.5 for p in prices],
            "Volume": [1_000_000] * 30,
        }, index=dates)
        result = mfi(df, 14)
        valid = result.dropna()
        # With all positive price movement, neg_sum = 0, ratio = inf,
        # MFI = 100 - (100 / (1 + inf)) = 100 - 0 = 100.
        # Or it could be NaN if neg_sum.replace(0, NaN) makes ratio NaN.
        for val in valid:
            assert pd.isna(val) or val > 90.0, (
                f"MFI with all positive movement should be near 100 or NaN, got {val}"
            )

    def test_sma_crossover_with_nan_gaps(self):
        """SMA crossover strategy handles NaN gaps mid-series gracefully."""
        s = get_strategy("sma_crossover")
        s.configure({"sma_fast": 5, "sma_slow": 10})

        df = make_price_df(days=50)
        # Insert NaN gap in the middle of the series
        df_with_gaps = df.copy()
        df_with_gaps.loc[df_with_gaps.index[20:23], "Close"] = float("nan")

        result = s.compute_indicators(df_with_gaps)
        assert "sma_fast" in result.columns
        assert "sma_slow" in result.columns
        # SMA values around the NaN gap should propagate NaN
        # (rolling window with NaN inside should produce NaN)
        assert pd.isna(result["sma_fast"].iloc[22])  # within the gap
        # Values before the gap should still be valid
        assert pd.notna(result["sma_fast"].iloc[15])
        # Values after gap + window size should recover
        assert pd.notna(result["sma_fast"].iloc[30])
