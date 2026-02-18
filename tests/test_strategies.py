"""Tests for strategy framework and SMA crossover."""

import math
import pandas as pd
import pytest
from datetime import date

from backtester.strategies.registry import get_strategy, list_strategies
from backtester.strategies.indicators import sma, rsi, macd
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
