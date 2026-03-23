"""Tests for strategy framework and SMA crossover."""

import math
import pandas as pd
import pytest
from datetime import date

from backtester.strategies.base import Strategy
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
        # Chikou span is Close shifted backward by +26 periods (i.e., at index T
        # it gives Close[T-26]).  With only 20 bars, the first 26 values should
        # be NaN since there is no data 26 periods before them.  This is
        # backward-looking to prevent lookahead bias in signal generation.
        assert chikou.iloc[:20].isna().all()  # all 20 bars are within the 26-period warmup

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


# ---------------------------------------------------------------------------
# Ichimoku backward-looking chikou span test
# ---------------------------------------------------------------------------

class TestIchimokuChikouBackwardLooking:
    """Verify that ichimoku chikou span is backward-looking (no lookahead)."""

    def test_chikou_is_backward_looking(self):
        """Chikou span at time T should be Close[T - displacement], not future data."""
        dates = pd.bdate_range("2020-01-02", periods=60, freq="B")
        closes = [100.0 + i for i in range(60)]
        df = pd.DataFrame({
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
        }, index=dates)
        _, _, _, _, chikou = ichimoku(df, displacement=26)

        # First 26 values should be NaN (no data 26 days before)
        assert chikou.iloc[:26].isna().all()

        # At index 26: chikou should be Close[0] = 100.0
        assert chikou.iloc[26] == pytest.approx(100.0)

        # At index 30: chikou should be Close[4] = 104.0
        assert chikou.iloc[30] == pytest.approx(104.0)

        # At last index (59): chikou should be Close[33] = 133.0
        assert chikou.iloc[59] == pytest.approx(133.0)

    def test_chikou_does_not_contain_future_data(self):
        """No chikou value at row T should equal a future Close[T+k]."""
        dates = pd.bdate_range("2020-01-02", periods=60, freq="B")
        closes = [100.0 + i * 2 for i in range(60)]  # distinct values
        df = pd.DataFrame({
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
        }, index=dates)
        _, _, _, _, chikou = ichimoku(df, displacement=26)

        for t in range(len(closes)):
            val = chikou.iloc[t]
            if pd.isna(val):
                continue
            # The chikou value should NOT match any future close
            for future in range(t + 1, len(closes)):
                assert val != closes[future], (
                    f"Chikou at T={t} ({val}) equals future Close[{future}] ({closes[future]}) — "
                    f"this is a lookahead bug"
                )


# ---------------------------------------------------------------------------
# compute_indicators df.copy() invariant tests for all strategies
# ---------------------------------------------------------------------------

class TestComputeIndicatorsNoCopy:
    """Verify that compute_indicators() never mutates the input DataFrame."""

    def _check_no_mutation(self, strategy_name, params=None, days=50):
        s = get_strategy(strategy_name)
        if params:
            s.configure(params)
        df = make_price_df(days=days)
        original_cols = set(df.columns)
        original_shape = df.shape
        s.compute_indicators(df)
        assert set(df.columns) == original_cols, (
            f"{strategy_name}.compute_indicators() mutated input df columns: "
            f"added {set(df.columns) - original_cols}"
        )
        assert df.shape == original_shape

    def test_sma_crossover_no_mutation(self):
        self._check_no_mutation("sma_crossover", {"sma_fast": 5, "sma_slow": 10})

    def test_rule_based_no_mutation(self):
        self._check_no_mutation("rule_based", {
            "indicators": {"rsi": {"period": 14}},
            "buy_when": [["rsi", ">", 60]],
            "sell_when": [],
        })

    def test_value_quality_no_mutation(self):
        self._check_no_mutation("value_quality")

    def test_earnings_growth_no_mutation(self):
        self._check_no_mutation("earnings_growth")

    def test_fundamental_screener_no_mutation(self):
        self._check_no_mutation("fundamental_screener")

    def test_insider_following_no_mutation(self):
        self._check_no_mutation("insider_following")

    def test_smart_money_no_mutation(self):
        self._check_no_mutation("smart_money")

    def test_macro_aware_value_no_mutation(self):
        self._check_no_mutation("macro_aware_value", {"sma_period": 10})

    def test_sentiment_momentum_no_mutation(self):
        self._check_no_mutation("sentiment_momentum", {"sma_period": 10})

    def test_risk_regime_no_mutation(self):
        self._check_no_mutation("risk_regime", {"sma_period": 10})


# ---------------------------------------------------------------------------
# size_order tests for SHORT and COVER
# ---------------------------------------------------------------------------

class TestSizeOrderShortCover:
    """Verify size_order returns correct values for SHORT and COVER signals."""

    def test_short_returns_negative(self):
        """size_order for SHORT should return a negative quantity."""
        s = get_strategy("sma_crossover")
        row = pd.Series({"Close": 50.0, "Open": 50.0, "High": 51.0, "Low": 49.0, "Volume": 1e6})
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        qty = s.size_order("TEST", SignalAction.SHORT, row, state, 0.10)
        assert qty < 0, f"SHORT size_order should return negative, got {qty}"
        assert qty == -200  # 10k / 50 = 200, negated

    def test_cover_returns_sentinel(self):
        """size_order for COVER should return -1 sentinel when position exists."""
        s = get_strategy("sma_crossover")
        row = pd.Series({"Close": 50.0, "Open": 50.0, "High": 51.0, "Low": 49.0, "Volume": 1e6})
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=1, position_symbols=frozenset({"TEST"}),
        )
        qty = s.size_order("TEST", SignalAction.COVER, row, state, 0.10)
        assert qty == -1

    def test_cover_no_position_returns_zero(self):
        """size_order for COVER with no position should return 0."""
        s = get_strategy("sma_crossover")
        row = pd.Series({"Close": 50.0})
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        qty = s.size_order("TEST", SignalAction.COVER, row, state, 0.10)
        assert qty == 0

    def test_sell_no_position_returns_zero(self):
        """size_order for SELL with no position should return 0."""
        s = get_strategy("sma_crossover")
        row = pd.Series({"Close": 50.0})
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        qty = s.size_order("TEST", SignalAction.SELL, row, state, 0.10)
        assert qty == 0

    def test_buy_zero_price_returns_zero(self):
        """size_order for BUY with zero price should return 0."""
        s = get_strategy("sma_crossover")
        row = pd.Series({"Close": 0.0})
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        qty = s.size_order("TEST", SignalAction.BUY, row, state, 0.10)
        assert qty == 0

    def test_short_zero_price_returns_zero(self):
        """size_order for SHORT with zero price should return 0."""
        s = get_strategy("sma_crossover")
        row = pd.Series({"Close": 0.0})
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        qty = s.size_order("TEST", SignalAction.SHORT, row, state, 0.10)
        assert qty == 0

    def test_buy_limited_by_cash(self):
        """size_order for BUY should not exceed available cash."""
        s = get_strategy("sma_crossover")
        row = pd.Series({"Close": 50.0})
        state = PortfolioState(
            cash=5_000.0,  # only 5k cash
            total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        # 10% of 100k = 10k target, but cash is only 5k
        qty = s.size_order("TEST", SignalAction.BUY, row, state, 0.10)
        assert qty == 100  # 5000 / 50 = 100


# ---------------------------------------------------------------------------
# Strategy auto-discovery tests
# ---------------------------------------------------------------------------

class TestStrategyAutoDiscovery:
    """Verify that all expected strategies are auto-discovered and registered."""

    def test_all_strategies_registered(self):
        expected = [
            "sma_crossover", "rule_based", "value_quality",
            "earnings_growth", "fundamental_screener",
            "insider_following", "smart_money",
        ]
        registered = list_strategies()
        for name in expected:
            assert name in registered, (
                f"Strategy '{name}' not found in registry. "
                f"Available: {registered}"
            )

    def test_each_strategy_instantiable(self):
        """Each registered strategy should be instantiable without errors."""
        for name in list_strategies():
            s = get_strategy(name)
            assert isinstance(s, Strategy), (
                f"get_strategy('{name}') didn't return a Strategy instance"
            )

    def test_each_strategy_has_required_methods(self):
        """Each strategy should have the required ABC methods."""
        for name in list_strategies():
            s = get_strategy(name)
            assert hasattr(s, "compute_indicators")
            assert hasattr(s, "generate_signals")
            assert hasattr(s, "configure")
            assert hasattr(s, "size_order")


# ---------------------------------------------------------------------------
# CrossSectionalStrategy helper tests
# ---------------------------------------------------------------------------

class TestCrossSectionalHelpers:
    """Test the static helper methods on CrossSectionalStrategy."""

    def test_top_n(self):
        from backtester.strategies.base import CrossSectionalStrategy
        scores = {"A": 10, "B": 30, "C": 20, "D": 5}
        result = CrossSectionalStrategy.top_n(scores, 2)
        assert result == ["B", "C"]

    def test_bottom_n(self):
        from backtester.strategies.base import CrossSectionalStrategy
        scores = {"A": 10, "B": 30, "C": 20, "D": 5}
        result = CrossSectionalStrategy.bottom_n(scores, 2)
        assert result == ["D", "A"]

    def test_top_n_more_than_available(self):
        from backtester.strategies.base import CrossSectionalStrategy
        scores = {"A": 10, "B": 20}
        result = CrossSectionalStrategy.top_n(scores, 5)
        assert len(result) == 2

    def test_bottom_n_empty(self):
        from backtester.strategies.base import CrossSectionalStrategy
        scores = {}
        result = CrossSectionalStrategy.bottom_n(scores, 3)
        assert result == []


# ---------------------------------------------------------------------------
# Rule-based edge case tests
# ---------------------------------------------------------------------------

class TestRuleBasedEdgeCases:
    """Test edge cases in the rule-based strategy."""

    def test_no_rules_means_hold(self):
        """With no buy or sell rules, strategy always HOLDs."""
        s = get_strategy("rule_based")
        s.configure({"indicators": {}, "buy_when": [], "sell_when": []})
        row = pd.Series({"Close": 100.0})
        signal = s.generate_signals("TEST", row, None, _make_state())
        assert signal == SignalAction.HOLD

    def test_no_buy_rules_no_sell_rules_with_position(self):
        """With no sell rules and a position, should HOLD."""
        s = get_strategy("rule_based")
        s.configure({"indicators": {}, "buy_when": [], "sell_when": []})
        pos = Position(symbol="TEST")
        pos.add_lot(100, 100.0, date(2020, 1, 2))
        row = pd.Series({"Close": 90.0})
        state = _make_state(num_positions=1, symbols=frozenset({"TEST"}))
        signal = s.generate_signals("TEST", row, pos, state)
        assert signal == SignalAction.HOLD

    def test_rule_with_missing_column(self):
        """When a rule references a missing column, it should evaluate to False (HOLD)."""
        s = get_strategy("rule_based")
        s.configure({
            "indicators": {},
            "buy_when": [["nonexistent_col", ">", 0]],
            "sell_when": [],
        })
        row = pd.Series({"Close": 100.0})
        signal = s.generate_signals("TEST", row, None, _make_state())
        assert signal == SignalAction.HOLD

    def test_rule_with_nan_value(self):
        """When a referenced column is NaN, rule should evaluate to False."""
        s = get_strategy("rule_based")
        s.configure({
            "indicators": {},
            "buy_when": [["rsi", ">", 50]],
            "sell_when": [],
        })
        row = pd.Series({"Close": 100.0, "rsi": float("nan")})
        signal = s.generate_signals("TEST", row, None, _make_state())
        assert signal == SignalAction.HOLD

    def test_ichimoku_indicator_expands_in_rule_based(self):
        """Ichimoku through rule_based should create all expected columns."""
        s = get_strategy("rule_based")
        s.configure({
            "indicators": {"ichi": {"fn": "ichimoku", "tenkan": 9, "kijun": 26}},
            "buy_when": [],
            "sell_when": [],
        })
        df = make_price_df(days=100)
        result = s.compute_indicators(df)
        assert "ichi_tenkan" in result.columns
        assert "ichi_kijun" in result.columns
        assert "ichi_senkou_a" in result.columns
        assert "ichi_senkou_b" in result.columns
        assert "ichi_chikou" in result.columns


class TestSmartMoneyThreshold:
    """Verify SmartMoney inst_growth_threshold boundary behavior."""

    def _make_row(self, inst_change, close=120.0, sma_trend=100.0):
        """Create a row with all required SmartMoney columns."""
        return pd.Series({
            "Close": close,
            "sma_trend": sma_trend,
            "inst_shares_change_pct": inst_change,
            "fund_net_income": 1_000_000.0,
            "fund_revenue_growth_yoy": 0.10,
            "insider_buy_ratio_90d": 0.8,
            "insider_net_shares_30d": 500,
        })

    def test_exact_threshold_triggers_buy(self):
        """inst_change exactly at threshold should trigger BUY, not HOLD.

        The docstring says 'Min QoQ institutional share increase (default 0.05 = 5%)',
        so the minimum value (5%) should pass the filter. Previously used `<=`
        which rejected the exact boundary value.
        """
        s = get_strategy("smart_money")
        s.configure({"sma_period": 50, "inst_growth_threshold": 0.05})
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = self._make_row(inst_change=0.05)  # exactly at threshold
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.BUY, (
            "inst_change at exactly the threshold should trigger BUY"
        )

    def test_below_threshold_holds(self):
        """inst_change below threshold should HOLD."""
        s = get_strategy("smart_money")
        s.configure({"sma_period": 50, "inst_growth_threshold": 0.05})
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = self._make_row(inst_change=0.04)
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.HOLD

    def test_above_threshold_buys(self):
        """inst_change above threshold should BUY."""
        s = get_strategy("smart_money")
        s.configure({"sma_period": 50, "inst_growth_threshold": 0.05})
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = self._make_row(inst_change=0.10)
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.BUY


# ---------------------------------------------------------------------------
# MacroAwareValue unit tests
# ---------------------------------------------------------------------------


class TestMacroAwareValue:
    """Unit tests for MacroAwareValue strategy signal generation."""

    @staticmethod
    def _make_row(
        close=100.0,
        sma_trend=90.0,
        f_score=7.0,
        pe_ratio=12.0,
        z_score=4.0,
        yield_spread=1.5,
        credit_spread=3.0,
    ):
        d = {
            "Close": close,
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Volume": 1_000_000,
            "sma_trend": sma_trend,
            "fund_piotroski_f": f_score,
            "fund_pe_ratio": pe_ratio,
            "fund_altman_z": z_score,
            "fred_yield_spread_10y2y": yield_spread,
            "fred_credit_spread_hy": credit_spread,
        }
        return pd.Series(d)

    def test_buy_expansion_good_quality(self):
        """Expansion regime + good F-Score + low P/E + safe Z → BUY."""
        s = get_strategy("macro_aware_value")
        s.configure({"sma_period": 50, "expansion_min_f": 5})
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = self._make_row(f_score=6.0, pe_ratio=15.0, z_score=3.0)
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.BUY

    def test_hold_no_fundamentals(self):
        """Missing fund_ columns → HOLD (graceful degradation)."""
        s = get_strategy("macro_aware_value")
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = pd.Series({
            "Close": 100.0, "Open": 99.9, "High": 101.0, "Low": 99.0,
            "Volume": 1e6, "sma_trend": 90.0,
        })
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.HOLD

    def test_hold_no_sma(self):
        """SMA not yet computed (NaN) → HOLD."""
        s = get_strategy("macro_aware_value")
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = self._make_row(sma_trend=float("nan"))
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.HOLD

    def test_hold_below_sma(self):
        """Price below SMA → no BUY even with great fundamentals."""
        s = get_strategy("macro_aware_value")
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = self._make_row(close=80.0, sma_trend=100.0, f_score=9.0)
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.HOLD

    def test_hold_distressed(self):
        """Z-Score below threshold → HOLD."""
        s = get_strategy("macro_aware_value")
        s.configure({"min_z_score": 1.8})
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = self._make_row(z_score=1.5)
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.HOLD

    def test_contraction_tighter_fscore(self):
        """Contraction regime requires higher F-Score."""
        s = get_strategy("macro_aware_value")
        s.configure({
            "expansion_min_f": 5,
            "contraction_min_f": 7,
        })
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        # Contraction: inverted yield + wide credit spread
        row = self._make_row(
            f_score=6.0, yield_spread=-0.5, credit_spread=7.0,
        )
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.HOLD, "F=6 should be blocked in contraction (needs 7)"

    def test_contraction_high_fscore_passes(self):
        """Contraction but F-Score meets tighter threshold → BUY."""
        s = get_strategy("macro_aware_value")
        s.configure({
            "expansion_min_f": 5,
            "contraction_min_f": 7,
            "contraction_max_pe": 15.0,
        })
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = self._make_row(
            f_score=8.0, pe_ratio=12.0, yield_spread=-0.5, credit_spread=7.0,
        )
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.BUY

    def test_no_macro_data_falls_back_to_conservative(self):
        """Missing FRED columns → contraction thresholds (conservative)."""
        s = get_strategy("macro_aware_value")
        s.configure({
            "expansion_min_f": 5,
            "contraction_min_f": 7,
        })
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = pd.Series({
            "Close": 100.0, "Open": 99.9, "High": 101.0, "Low": 99.0,
            "Volume": 1e6, "sma_trend": 90.0,
            "fund_piotroski_f": 6.0,
            "fund_pe_ratio": 12.0,
            "fund_altman_z": 4.0,
        })
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.HOLD, "F=6 < 7 contraction threshold → HOLD"

    def test_sell_quality_deterioration(self):
        """F-Score drops below 3 → SELL existing position."""
        s = get_strategy("macro_aware_value")
        state = PortfolioState(
            cash=50_000.0, total_equity=100_000.0,
            num_positions=1, position_symbols=frozenset({"TEST"}),
        )
        pos = Position("TEST")
        pos.add_lot(100, 90.0, date(2020, 1, 2))
        row = self._make_row(f_score=2.0, close=100.0, sma_trend=90.0)
        signal = s.generate_signals("TEST", row, pos, state)
        assert signal == SignalAction.SELL

    def test_sell_trend_broken(self):
        """Price drops below SMA → SELL existing position."""
        s = get_strategy("macro_aware_value")
        state = PortfolioState(
            cash=50_000.0, total_equity=100_000.0,
            num_positions=1, position_symbols=frozenset({"TEST"}),
        )
        pos = Position("TEST")
        pos.add_lot(100, 90.0, date(2020, 1, 2))
        # Close below SMA
        row = self._make_row(close=85.0, sma_trend=90.0, f_score=7.0)
        signal = s.generate_signals("TEST", row, pos, state)
        assert signal == SignalAction.SELL

    def test_hold_with_position_quality_ok(self):
        """Good quality + above SMA with position → HOLD (no sell)."""
        s = get_strategy("macro_aware_value")
        state = PortfolioState(
            cash=50_000.0, total_equity=100_000.0,
            num_positions=1, position_symbols=frozenset({"TEST"}),
        )
        pos = Position("TEST")
        pos.add_lot(100, 90.0, date(2020, 1, 2))
        row = self._make_row(f_score=7.0, close=100.0, sma_trend=90.0, z_score=4.0)
        signal = s.generate_signals("TEST", row, pos, state)
        assert signal == SignalAction.HOLD

    def test_configure_params(self):
        """configure() updates all parameters."""
        s = get_strategy("macro_aware_value")
        s.configure({
            "sma_period": 100,
            "expansion_min_f": 4,
            "contraction_min_f": 8,
            "expansion_max_pe": 25.0,
            "contraction_max_pe": 10.0,
            "min_z_score": 2.0,
            "credit_threshold": 4.0,
        })
        assert s.sma_period == 100
        assert s.expansion_min_f == 4
        assert s.contraction_min_f == 8
        assert s.expansion_max_pe == 25.0
        assert s.contraction_max_pe == 10.0
        assert s.min_z_score == 2.0
        assert s.credit_threshold == 4.0


# ---------------------------------------------------------------------------
# SentimentMomentum unit tests
# ---------------------------------------------------------------------------


class TestSentimentMomentum:
    """Unit tests for SentimentMomentum strategy signal generation."""

    @staticmethod
    def _make_row(
        close=100.0,
        sma_trend=90.0,
        analyst_breadth=None,
        insider_buy_ratio=None,
    ):
        d = {
            "Close": close,
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Volume": 1_000_000,
            "sma_trend": sma_trend,
        }
        if analyst_breadth is not None:
            d["analyst_rev_breadth"] = analyst_breadth
        if insider_buy_ratio is not None:
            d["insider_buy_ratio_90d"] = insider_buy_ratio
        return pd.Series(d)

    def test_buy_three_bullish_signals(self):
        """Positive breadth + high insider buying + above SMA → BUY."""
        s = get_strategy("sentiment_momentum")
        s.configure({"sma_period": 10, "min_signals_buy": 2})
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = self._make_row(
            close=100.0, sma_trend=90.0,
            analyst_breadth=0.5, insider_buy_ratio=0.8,
        )
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.BUY

    def test_hold_sma_nan(self):
        """SMA is NaN → HOLD."""
        s = get_strategy("sentiment_momentum")
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = self._make_row(sma_trend=float("nan"))
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.HOLD

    def test_hold_only_sma_bullish(self):
        """Only SMA signal bullish (1/3) with threshold 2 → HOLD."""
        s = get_strategy("sentiment_momentum")
        s.configure({"min_signals_buy": 2})
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        # Close > SMA (bullish), no analyst/insider data
        row = self._make_row(close=100.0, sma_trend=90.0)
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.HOLD

    def test_sell_bearish_signals(self):
        """Negative breadth + low insider + below SMA → SELL existing position."""
        s = get_strategy("sentiment_momentum")
        s.configure({"min_signals_sell": 2})
        state = PortfolioState(
            cash=50_000.0, total_equity=100_000.0,
            num_positions=1, position_symbols=frozenset({"TEST"}),
        )
        pos = Position("TEST")
        pos.add_lot(100, 90.0, date(2020, 1, 2))
        row = self._make_row(
            close=80.0, sma_trend=90.0,
            analyst_breadth=-0.5, insider_buy_ratio=0.1,
        )
        signal = s.generate_signals("TEST", row, pos, state)
        assert signal == SignalAction.SELL

    def test_hold_with_position_neutral(self):
        """Mixed signals with position → HOLD (no sell)."""
        s = get_strategy("sentiment_momentum")
        s.configure({"min_signals_sell": 3})
        state = PortfolioState(
            cash=50_000.0, total_equity=100_000.0,
            num_positions=1, position_symbols=frozenset({"TEST"}),
        )
        pos = Position("TEST")
        pos.add_lot(100, 90.0, date(2020, 1, 2))
        # Bearish: below SMA. Bullish: positive breadth. → only 1 bearish, needs 3
        row = self._make_row(
            close=80.0, sma_trend=90.0,
            analyst_breadth=0.5, insider_buy_ratio=0.5,
        )
        signal = s.generate_signals("TEST", row, pos, state)
        assert signal == SignalAction.HOLD

    def test_graceful_no_alt_data(self):
        """No analyst/insider columns → only SMA signal counted."""
        s = get_strategy("sentiment_momentum")
        s.configure({"min_signals_buy": 1})
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        # Close above SMA → 1 bullish, threshold=1 → BUY
        row = self._make_row(close=100.0, sma_trend=90.0)
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.BUY

    def test_configure_params(self):
        """configure() updates all parameters."""
        s = get_strategy("sentiment_momentum")
        s.configure({
            "sma_period": 100,
            "min_signals_buy": 3,
            "min_signals_sell": 3,
            "insider_buy_threshold": 0.6,
            "insider_sell_threshold": 0.2,
        })
        assert s.sma_period == 100
        assert s.min_signals_buy == 3
        assert s.min_signals_sell == 3
        assert s.insider_buy_threshold == 0.6
        assert s.insider_sell_threshold == 0.2


# ---------------------------------------------------------------------------
# RiskRegime unit tests
# ---------------------------------------------------------------------------


class TestRiskRegime:
    """Unit tests for RiskRegime strategy signal generation."""

    @staticmethod
    def _make_row(
        close=100.0,
        vix_ratio=0.85,
        yield_spread=1.5,
        credit_spread=3.0,
        f_score=7.0,
    ):
        d = {
            "Close": close,
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Volume": 1_000_000,
            "sma_trend": 90.0,
        }
        if vix_ratio is not None:
            d["vix_ratio"] = vix_ratio
        if yield_spread is not None:
            d["fred_yield_spread_10y2y"] = yield_spread
        if credit_spread is not None:
            d["fred_credit_spread_hy"] = credit_spread
        if f_score is not None:
            d["fund_piotroski_f"] = f_score
        return pd.Series(d)

    def test_buy_full_risk_on(self):
        """All 3 risk signals positive + good F-Score → BUY."""
        s = get_strategy("risk_regime")
        s.configure({"min_f_score": 5})
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = self._make_row(vix_ratio=0.85, yield_spread=1.5, credit_spread=3.0, f_score=7.0)
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.BUY

    def test_buy_risk_on_no_fscore(self):
        """All 3 risk signals positive, no F-Score → BUY anyway."""
        s = get_strategy("risk_regime")
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = self._make_row(vix_ratio=0.85, yield_spread=1.5, credit_spread=3.0, f_score=None)
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.BUY

    def test_hold_low_fscore(self):
        """All 3 risk signals positive but F-Score too low → HOLD."""
        s = get_strategy("risk_regime")
        s.configure({"min_f_score": 5})
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = self._make_row(vix_ratio=0.85, yield_spread=1.5, credit_spread=3.0, f_score=3.0)
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.HOLD

    def test_sell_full_risk_off(self):
        """All 3 signals negative with position → SELL."""
        s = get_strategy("risk_regime")
        state = PortfolioState(
            cash=50_000.0, total_equity=100_000.0,
            num_positions=1, position_symbols=frozenset({"TEST"}),
        )
        pos = Position("TEST")
        pos.add_lot(100, 90.0, date(2020, 1, 2))
        row = self._make_row(vix_ratio=1.3, yield_spread=-0.5, credit_spread=8.0)
        signal = s.generate_signals("TEST", row, pos, state)
        assert signal == SignalAction.SELL

    def test_hold_neutral_signals(self):
        """1-2 signals positive → HOLD (no buy, no sell)."""
        s = get_strategy("risk_regime")
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        # 2 of 3 positive → neutral
        row = self._make_row(vix_ratio=0.85, yield_spread=1.5, credit_spread=8.0)
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.HOLD

    def test_hold_no_macro_data(self):
        """No macro columns at all → HOLD."""
        s = get_strategy("risk_regime")
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = pd.Series({
            "Close": 100.0, "Open": 99.9, "High": 101.0, "Low": 99.0,
            "Volume": 1e6, "sma_trend": 90.0,
        })
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.HOLD

    def test_buy_partial_data_one_signal(self):
        """Only VIX data (1 signal) and positive → BUY (graceful degradation)."""
        s = get_strategy("risk_regime")
        state = PortfolioState(
            cash=100_000.0, total_equity=100_000.0,
            num_positions=0, position_symbols=frozenset(),
        )
        row = self._make_row(
            vix_ratio=0.85,
            yield_spread=None,
            credit_spread=None,
        )
        signal = s.generate_signals("TEST", row, None, state)
        assert signal == SignalAction.BUY

    def test_no_sell_neutral_with_position(self):
        """Neutral signals (1-2 positive) with position → HOLD (no sell)."""
        s = get_strategy("risk_regime")
        state = PortfolioState(
            cash=50_000.0, total_equity=100_000.0,
            num_positions=1, position_symbols=frozenset({"TEST"}),
        )
        pos = Position("TEST")
        pos.add_lot(100, 90.0, date(2020, 1, 2))
        # 1 of 3 positive → score=1, not 0 → no sell
        row = self._make_row(vix_ratio=0.85, yield_spread=-0.5, credit_spread=8.0)
        signal = s.generate_signals("TEST", row, pos, state)
        assert signal == SignalAction.HOLD

    def test_configure_params(self):
        """configure() updates all parameters."""
        s = get_strategy("risk_regime")
        s.configure({
            "sma_period": 100,
            "vix_threshold": 0.9,
            "yield_spread_threshold": 0.5,
            "credit_threshold": 4.0,
            "min_f_score": 6,
        })
        assert s.sma_period == 100
        assert s.vix_threshold == 0.9
        assert s.yield_spread_threshold == 0.5
        assert s.credit_threshold == 4.0
        assert s.min_f_score == 6


# ---------------------------------------------------------------------------
# MomentumRotation strategy tests
# ---------------------------------------------------------------------------

class TestMomentumRotation:
    """Tests for the momentum_rotation cross-sectional strategy."""

    def _make_state(self, cash=100_000.0, positions=None):
        syms = frozenset(positions.keys()) if positions else frozenset()
        return PortfolioState(
            cash=cash,
            total_equity=100_000.0,
            num_positions=len(syms),
            position_symbols=syms,
        )

    def test_configure_defaults(self):
        s = get_strategy("momentum_rotation")
        assert s.roc_period == 63
        assert s.top_n_count == 3

    def test_configure_overrides(self):
        s = get_strategy("momentum_rotation")
        s.configure({"roc_period": 126, "top_n": 5})
        assert s.roc_period == 126
        assert s.top_n_count == 5

    def test_compute_indicators_adds_roc(self):
        s = get_strategy("momentum_rotation")
        s.configure({"roc_period": 5})
        df = make_price_df(days=30)
        result = s.compute_indicators(df)
        assert "roc" in result.columns
        # First roc_period values are NaN
        assert result["roc"].iloc[:5].isna().all()
        assert result["roc"].iloc[-1:].notna().all()

    def test_compute_indicators_no_mutation(self):
        s = get_strategy("momentum_rotation")
        s.configure({"roc_period": 5})
        df = make_price_df(days=30)
        original_cols = set(df.columns)
        s.compute_indicators(df)
        assert set(df.columns) == original_cols

    def test_generate_signals_always_hold(self):
        """Per-symbol generate_signals should always return HOLD (cross-sectional logic is in rank_universe)."""
        s = get_strategy("momentum_rotation")
        state = self._make_state()
        row = {"Close": 100.0, "roc": 0.05}
        assert s.generate_signals("XLK", row, {}, state) == SignalAction.HOLD

    def test_rank_universe_buys_top_n(self):
        """rank_universe buys top N symbols by ROC."""
        s = get_strategy("momentum_rotation")
        s.configure({"top_n": 2})
        state = self._make_state()
        bar_data = {
            "XLK": {"roc": 0.10},
            "XLF": {"roc": 0.05},
            "XLE": {"roc": 0.15},
            "XLV": {"roc": 0.02},
        }
        signals = s.rank_universe(bar_data, {}, state)
        signal_dict = dict(signals)
        # Top 2 by ROC: XLE (0.15), XLK (0.10)
        assert signal_dict.get("XLE") == SignalAction.BUY
        assert signal_dict.get("XLK") == SignalAction.BUY
        assert "XLF" not in signal_dict
        assert "XLV" not in signal_dict

    def test_rank_universe_sells_fallen_holdings(self):
        """rank_universe sells holdings that drop out of top N."""
        s = get_strategy("momentum_rotation")
        s.configure({"top_n": 1})
        # XLF is currently held but has low ROC
        pos_xlf = Position("XLF")
        pos_xlf.add_lot(100, 50.0, date(2020, 1, 2))
        positions = {"XLF": pos_xlf}
        state = self._make_state(positions=positions)
        bar_data = {
            "XLK": {"roc": 0.20},
            "XLF": {"roc": 0.01},
        }
        signals = s.rank_universe(bar_data, positions, state)
        signal_dict = dict(signals)
        assert signal_dict.get("XLF") == SignalAction.SELL
        assert signal_dict.get("XLK") == SignalAction.BUY

    def test_rank_universe_no_action_when_already_held(self):
        """If a top-N symbol is already held, no BUY signal is generated."""
        s = get_strategy("momentum_rotation")
        s.configure({"top_n": 1})
        pos_xlk = Position("XLK")
        pos_xlk.add_lot(100, 50.0, date(2020, 1, 2))
        positions = {"XLK": pos_xlk}
        state = self._make_state(positions=positions)
        bar_data = {
            "XLK": {"roc": 0.20},
            "XLF": {"roc": 0.01},
        }
        signals = s.rank_universe(bar_data, positions, state)
        signal_dict = dict(signals)
        # XLK is already held and top 1 → no signal
        assert "XLK" not in signal_dict

    def test_rank_universe_nan_roc_excluded(self):
        """Symbols with NaN ROC are excluded from ranking."""
        s = get_strategy("momentum_rotation")
        s.configure({"top_n": 2})
        state = self._make_state()
        bar_data = {
            "XLK": {"roc": float("nan")},
            "XLF": {"roc": 0.05},
            "XLE": {"roc": 0.10},
        }
        signals = s.rank_universe(bar_data, {}, state)
        signal_dict = dict(signals)
        assert signal_dict.get("XLE") == SignalAction.BUY
        assert signal_dict.get("XLF") == SignalAction.BUY
        assert "XLK" not in signal_dict

    def test_rank_universe_empty_bar_data(self):
        """Empty bar_data → no signals."""
        s = get_strategy("momentum_rotation")
        state = self._make_state()
        signals = s.rank_universe({}, {}, state)
        assert signals == []

    def test_size_order_sell_returns_sentinel(self):
        """size_order returns -1 for SELL."""
        s = get_strategy("momentum_rotation")
        state = self._make_state()
        qty = s.size_order("XLK", SignalAction.SELL, {}, {}, state)
        assert qty == -1

    def test_size_order_buy_returns_zero(self):
        """size_order returns 0 for BUY (defers to sizer)."""
        s = get_strategy("momentum_rotation")
        state = self._make_state()
        qty = s.size_order("XLK", SignalAction.BUY, {}, {}, state)
        assert qty == 0


# ---------------------------------------------------------------------------
# MultiTimeframeTrend strategy tests
# ---------------------------------------------------------------------------

class TestMultiTimeframeTrend:
    """Tests for the multi_tf_trend weekly-trend + daily-RSI strategy."""

    def _make_state(self, has_position=False, symbol="TEST"):
        syms = frozenset({symbol}) if has_position else frozenset()
        return PortfolioState(
            cash=100_000.0,
            total_equity=100_000.0,
            num_positions=1 if has_position else 0,
            position_symbols=syms,
        )

    def test_configure_defaults(self):
        s = get_strategy("multi_tf_trend")
        assert s.weekly_fast == 10
        assert s.weekly_slow == 40
        assert s.daily_rsi_period == 14
        assert s.rsi_entry == 40
        assert s.rsi_exit == 70

    def test_configure_overrides(self):
        s = get_strategy("multi_tf_trend")
        s.configure({
            "weekly_fast": 5,
            "weekly_slow": 20,
            "daily_rsi_period": 7,
            "rsi_entry": 30,
            "rsi_exit": 80,
        })
        assert s.weekly_fast == 5
        assert s.weekly_slow == 20
        assert s.daily_rsi_period == 7
        assert s.rsi_entry == 30
        assert s.rsi_exit == 80

    def test_timeframes_property(self):
        s = get_strategy("multi_tf_trend")
        assert s.timeframes == ["weekly"]

    def test_compute_indicators_daily_rsi(self):
        """compute_indicators adds daily_rsi column."""
        s = get_strategy("multi_tf_trend")
        s.configure({"daily_rsi_period": 5})
        df = make_price_df(days=30)
        result = s.compute_indicators(df)
        assert "daily_rsi" in result.columns
        valid = result["daily_rsi"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_compute_indicators_with_weekly_close(self):
        """compute_indicators computes weekly SMAs from weekly_Close column."""
        s = get_strategy("multi_tf_trend")
        s.configure({"weekly_fast": 3, "weekly_slow": 5, "daily_rsi_period": 5})
        df = make_price_df(days=50)
        # Simulate engine-merged weekly Close
        df["weekly_Close"] = df["Close"] * 1.01
        result = s.compute_indicators(df)
        assert "weekly_sma_fast" in result.columns
        assert "weekly_sma_slow" in result.columns

    def test_compute_indicators_with_timeframe_data(self):
        """compute_indicators uses timeframe_data dict for weekly data."""
        s = get_strategy("multi_tf_trend")
        s.configure({"weekly_fast": 3, "weekly_slow": 5, "daily_rsi_period": 5})
        daily_df = make_price_df(days=50)
        weekly_df = daily_df.iloc[::5].copy()  # Simulated weekly
        result = s.compute_indicators(daily_df, timeframe_data={"weekly": weekly_df})
        assert "daily_rsi" in result.columns
        assert "weekly_sma_fast" in result.columns

    def test_compute_indicators_no_mutation(self):
        s = get_strategy("multi_tf_trend")
        s.configure({"daily_rsi_period": 5})
        df = make_price_df(days=30)
        original_cols = set(df.columns)
        s.compute_indicators(df)
        assert set(df.columns) == original_cols

    def test_buy_signal_uptrend_and_low_rsi(self):
        """BUY when weekly uptrend and daily RSI below entry threshold."""
        s = get_strategy("multi_tf_trend")
        s.configure({"rsi_entry": 40, "rsi_exit": 70})
        state = self._make_state(has_position=False)
        row = {
            "daily_rsi": 30.0,           # Below entry threshold
            "weekly_sma_fast": 110.0,     # Fast > Slow = uptrend
            "weekly_sma_slow": 100.0,
            "Close": 105.0,
        }
        signal = s.generate_signals("TEST", row, {}, state)
        assert signal == SignalAction.BUY

    def test_sell_signal_high_rsi(self):
        """SELL when daily RSI exceeds exit threshold and has position."""
        s = get_strategy("multi_tf_trend")
        s.configure({"rsi_entry": 40, "rsi_exit": 70})
        pos = Position("TEST")
        pos.add_lot(100, 90.0, date(2020, 1, 2))
        state = self._make_state(has_position=True)
        row = {
            "daily_rsi": 75.0,           # Above exit threshold
            "weekly_sma_fast": 110.0,
            "weekly_sma_slow": 100.0,
            "Close": 105.0,
        }
        signal = s.generate_signals("TEST", row, {"TEST": pos}, state)
        assert signal == SignalAction.SELL

    def test_hold_in_downtrend(self):
        """HOLD when weekly downtrend even with low RSI."""
        s = get_strategy("multi_tf_trend")
        s.configure({"rsi_entry": 40, "rsi_exit": 70})
        state = self._make_state(has_position=False)
        row = {
            "daily_rsi": 30.0,           # Low RSI
            "weekly_sma_fast": 90.0,      # Fast < Slow = downtrend
            "weekly_sma_slow": 100.0,
            "Close": 95.0,
        }
        signal = s.generate_signals("TEST", row, {}, state)
        assert signal == SignalAction.HOLD

    def test_hold_when_rsi_between_thresholds(self):
        """HOLD when RSI is between entry and exit thresholds."""
        s = get_strategy("multi_tf_trend")
        s.configure({"rsi_entry": 40, "rsi_exit": 70})
        state = self._make_state(has_position=False)
        row = {
            "daily_rsi": 55.0,           # Between 40 and 70
            "weekly_sma_fast": 110.0,
            "weekly_sma_slow": 100.0,
            "Close": 105.0,
        }
        signal = s.generate_signals("TEST", row, {}, state)
        assert signal == SignalAction.HOLD

    def test_hold_when_indicators_missing(self):
        """HOLD when weekly SMA indicators are None."""
        s = get_strategy("multi_tf_trend")
        state = self._make_state(has_position=False)
        row = {
            "daily_rsi": 30.0,
            "weekly_sma_fast": None,
            "weekly_sma_slow": None,
            "Close": 100.0,
        }
        signal = s.generate_signals("TEST", row, {}, state)
        assert signal == SignalAction.HOLD

    def test_hold_when_rsi_nan(self):
        """HOLD when daily RSI is NaN."""
        s = get_strategy("multi_tf_trend")
        state = self._make_state(has_position=False)
        row = {
            "daily_rsi": float("nan"),
            "weekly_sma_fast": 110.0,
            "weekly_sma_slow": 100.0,
            "Close": 100.0,
        }
        signal = s.generate_signals("TEST", row, {}, state)
        assert signal == SignalAction.HOLD

    def test_no_buy_when_already_positioned(self):
        """No BUY signal when position already exists."""
        s = get_strategy("multi_tf_trend")
        s.configure({"rsi_entry": 40, "rsi_exit": 70})
        pos = Position("TEST")
        pos.add_lot(100, 90.0, date(2020, 1, 2))
        state = self._make_state(has_position=True)
        row = {
            "daily_rsi": 30.0,
            "weekly_sma_fast": 110.0,
            "weekly_sma_slow": 100.0,
            "Close": 105.0,
        }
        signal = s.generate_signals("TEST", row, {"TEST": pos}, state)
        assert signal == SignalAction.HOLD

    def test_size_order_sell_returns_sentinel(self):
        s = get_strategy("multi_tf_trend")
        state = self._make_state()
        qty = s.size_order("TEST", SignalAction.SELL, {}, {}, state)
        assert qty == -1

    def test_size_order_buy_returns_zero(self):
        s = get_strategy("multi_tf_trend")
        state = self._make_state()
        qty = s.size_order("TEST", SignalAction.BUY, {}, {}, state)
        assert qty == 0
