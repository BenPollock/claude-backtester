"""Tests for position sizing models."""

import pandas as pd
import pytest

from backtester.execution.position_sizing import (
    FixedFractional,
    ATRSizer,
    KellyCriterionSizer,
    RiskParitySizer,
    VolatilityParity,
)


class TestFixedFractional:
    def setup_method(self):
        self.sizer = FixedFractional()
        self.row = pd.Series({"Close": 100.0})

    def test_basic_allocation(self):
        # 10% of 100k equity = $10k, at $100/share = 100 shares
        shares = self.sizer.compute("TEST", 100.0, self.row, 100_000, 100_000, 0.10)
        assert shares == 100

    def test_price_zero(self):
        shares = self.sizer.compute("TEST", 0.0, self.row, 100_000, 100_000, 0.10)
        assert shares == 0

    def test_price_negative(self):
        shares = self.sizer.compute("TEST", -5.0, self.row, 100_000, 100_000, 0.10)
        assert shares == 0

    def test_cash_limited(self):
        # Only $5k cash, but 10% of 100k equity = $10k target → capped to $5k
        shares = self.sizer.compute("TEST", 100.0, self.row, 100_000, 5_000, 0.10)
        assert shares == 50

    def test_fractional_shares_truncated(self):
        # $10k target at $33/share = 303.03 → 303 shares
        shares = self.sizer.compute("TEST", 33.0, self.row, 100_000, 100_000, 0.10)
        assert shares == 303


class TestATRSizer:
    def setup_method(self):
        self.sizer = ATRSizer(risk_pct=0.01, atr_multiple=2.0)

    def test_normal_atr_sizing(self):
        # Risk = 0.01 * 100_000 = $1000, ATR risk = 5.0 * 2.0 = $10/share → 100 shares
        row = pd.Series({"Close": 100.0, "ATR": 5.0})
        shares = self.sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        assert shares == 100

    def test_fallback_when_atr_missing(self):
        row = pd.Series({"Close": 100.0})
        shares = self.sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        # Falls back to FixedFractional: min(100k*0.10, 100k) // 100 = 100
        assert shares == 100

    def test_fallback_when_atr_nan(self):
        row = pd.Series({"Close": 100.0, "ATR": float("nan")})
        shares = self.sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        assert shares == 100

    def test_fallback_when_atr_zero(self):
        row = pd.Series({"Close": 100.0, "ATR": 0.0})
        shares = self.sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        assert shares == 100  # fallback

    def test_cash_cap(self):
        # ATR sizing wants 100 shares, but only $3k cash → 30 shares max
        row = pd.Series({"Close": 100.0, "ATR": 5.0})
        shares = self.sizer.compute("TEST", 100.0, row, 100_000, 3_000, 0.10)
        assert shares == 30

    def test_max_alloc_cap(self):
        # ATR sizing wants 100 shares ($10k), but max_alloc = 5% of 100k = $5k → 50 shares
        row = pd.Series({"Close": 100.0, "ATR": 5.0})
        shares = self.sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.05)
        assert shares == 50

    def test_price_zero(self):
        row = pd.Series({"Close": 100.0, "ATR": 5.0})
        shares = self.sizer.compute("TEST", 0.0, row, 100_000, 100_000, 0.10)
        assert shares == 0


class TestVolatilityParity:
    def setup_method(self):
        self.sizer = VolatilityParity(target_vol=0.10)

    def test_normal_vol_parity_sizing(self):
        # ATR=2, price=100 → daily_vol=0.02, annual_vol=0.02*sqrt(252)≈0.3175
        # target_value = 100k * (0.10 / 0.3175) ≈ $31.5k → 315 shares
        # But capped by max_alloc: 100k * 0.10 = $10k → 100 shares
        row = pd.Series({"Close": 100.0, "ATR": 2.0})
        shares = self.sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        assert shares == 100  # capped by max_alloc

    def test_high_vol_reduces_position(self):
        # ATR=20, price=100 → daily_vol=0.20, annual_vol≈3.175
        # target_value = 100k * (0.10 / 3.175) ≈ $3.15k → 31 shares
        row = pd.Series({"Close": 100.0, "ATR": 20.0})
        shares = self.sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        assert shares == 31

    def test_fallback_when_atr_missing(self):
        row = pd.Series({"Close": 100.0})
        shares = self.sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        # Falls back to FixedFractional
        assert shares == 100

    def test_fallback_when_atr_zero(self):
        row = pd.Series({"Close": 100.0, "ATR": 0.0})
        shares = self.sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        # Falls back to FixedFractional
        assert shares == 100

    def test_price_zero(self):
        row = pd.Series({"Close": 100.0, "ATR": 2.0})
        shares = self.sizer.compute("TEST", 0.0, row, 100_000, 100_000, 0.10)
        assert shares == 0


class TestKellyCriterionSizer:
    def setup_method(self):
        self.sizer = KellyCriterionSizer(fraction=0.5)

    def test_normal_kelly(self):
        # win_rate=0.6, payoff=2.0 => f* = (0.6*2 - 0.4)/2 = 0.4
        # half-kelly => alloc=0.2, target=100k*0.2=20k, shares=200
        # but capped by max_alloc: 100k*0.10=10k => 100 shares
        row = pd.Series({"Close": 100.0, "kelly_win_rate": 0.6, "kelly_payoff_ratio": 2.0})
        shares = self.sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        assert shares == 100  # capped by max_alloc

    def test_win_rate_above_one(self):
        """win_rate > 1.0 is mathematically invalid but the sizer should not crash."""
        row = pd.Series({"Close": 100.0, "kelly_win_rate": 1.5, "kelly_payoff_ratio": 2.0})
        shares = self.sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        # f* = (1.5*2 - (-0.5))/2 = (3 + 0.5)/2 = 1.75, half = 0.875
        # capped by max_alloc 0.10 => 100k*0.10=10k => 100 shares
        assert shares == 100

    def test_win_rate_below_zero(self):
        """win_rate < 0 produces negative f*, so sizer returns 0 shares."""
        row = pd.Series({"Close": 100.0, "kelly_win_rate": -0.5, "kelly_payoff_ratio": 2.0})
        shares = self.sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        # f* = (-0.5*2 - 1.5)/2 = (-1 - 1.5)/2 = -1.25 => f* <= 0 => 0
        assert shares == 0

    def test_negative_payoff_falls_back(self):
        """Negative payoff should trigger fallback to FixedFractional."""
        row = pd.Series({"Close": 100.0, "kelly_win_rate": 0.6, "kelly_payoff_ratio": -1.0})
        shares = self.sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        # payoff <= 0 => fallback: min(100k*0.10, 100k) // 100 = 100
        assert shares == 100

    def test_zero_payoff_falls_back(self):
        """Zero payoff should trigger fallback to FixedFractional."""
        row = pd.Series({"Close": 100.0, "kelly_win_rate": 0.6, "kelly_payoff_ratio": 0.0})
        shares = self.sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        assert shares == 100  # fallback


class TestATRSizerEdgeCases:
    def test_atr_very_close_to_zero(self):
        """ATR near machine epsilon should produce very large sizing, capped by max_alloc."""
        sizer = ATRSizer(risk_pct=0.01, atr_multiple=2.0)
        row = pd.Series({"Close": 100.0, "ATR": 1e-10})
        shares = sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        # risk_per_share = 1e-10 * 2 = 2e-10 => shares = 1000/2e-10 = huge
        # capped by max_alloc: min(10k, 100k)//100 = 100
        assert shares == 100


class TestRiskParitySizer:
    def test_normal_sizing(self):
        sizer = RiskParitySizer(target_vol=0.10)
        row = pd.Series({"Close": 100.0, "ATR": 2.0})
        shares = sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        assert shares > 0

    def test_zero_price(self):
        """Zero price should return 0 shares."""
        sizer = RiskParitySizer(target_vol=0.10)
        row = pd.Series({"Close": 100.0, "ATR": 2.0})
        shares = sizer.compute("TEST", 0.0, row, 100_000, 100_000, 0.10)
        assert shares == 0

    def test_atr_missing_fallback(self):
        """Missing ATR should fall back to FixedFractional."""
        sizer = RiskParitySizer(target_vol=0.10)
        row = pd.Series({"Close": 100.0})
        shares = sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        assert shares == 100  # fallback


class TestVolatilityParityLookback:
    def test_lookback_parameter_stored_but_unused(self):
        """VolatilityParity accepts a lookback parameter but doesn't use it in compute().

        The parameter is stored for future use but the current implementation
        uses ATR from the row directly, ignoring lookback.
        """
        sizer_short = VolatilityParity(target_vol=0.10, lookback=5)
        sizer_long = VolatilityParity(target_vol=0.10, lookback=100)
        row = pd.Series({"Close": 100.0, "ATR": 2.0})

        shares_short = sizer_short.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        shares_long = sizer_long.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)

        # Same result regardless of lookback, since it's not used in compute
        assert shares_short == shares_long


class TestAllSizersNegativeEquity:
    """All sizers should return 0 (not negative shares) when equity is negative."""

    def test_fixed_fractional_negative_equity(self):
        sizer = FixedFractional()
        row = pd.Series({"Close": 100.0})
        shares = sizer.compute("TEST", 100.0, row, -50_000, 10_000, 0.10)
        assert shares == 0

    def test_atr_sizer_negative_equity(self):
        sizer = ATRSizer(risk_pct=0.01, atr_multiple=2.0)
        row = pd.Series({"Close": 100.0, "ATR": 5.0})
        shares = sizer.compute("TEST", 100.0, row, -50_000, 10_000, 0.10)
        assert shares == 0

    def test_kelly_sizer_negative_equity(self):
        sizer = KellyCriterionSizer(fraction=0.5)
        row = pd.Series({"Close": 100.0, "kelly_win_rate": 0.6, "kelly_payoff_ratio": 2.0})
        shares = sizer.compute("TEST", 100.0, row, -50_000, 10_000, 0.10)
        assert shares == 0

    def test_risk_parity_sizer_negative_equity(self):
        sizer = RiskParitySizer(target_vol=0.10)
        row = pd.Series({"Close": 100.0, "ATR": 2.0})
        shares = sizer.compute("TEST", 100.0, row, -50_000, 10_000, 0.10)
        assert shares == 0

    def test_volatility_parity_negative_equity(self):
        sizer = VolatilityParity(target_vol=0.10)
        row = pd.Series({"Close": 100.0, "ATR": 2.0})
        shares = sizer.compute("TEST", 100.0, row, -50_000, 10_000, 0.10)
        assert shares == 0


class TestAllSizersCashZero:
    """All sizers should return 0 when cash is 0."""

    def test_fixed_fractional_zero_cash(self):
        sizer = FixedFractional()
        row = pd.Series({"Close": 100.0})
        shares = sizer.compute("TEST", 100.0, row, 100_000, 0, 0.10)
        assert shares == 0

    def test_atr_sizer_zero_cash(self):
        sizer = ATRSizer(risk_pct=0.01, atr_multiple=2.0)
        row = pd.Series({"Close": 100.0, "ATR": 5.0})
        shares = sizer.compute("TEST", 100.0, row, 100_000, 0, 0.10)
        assert shares == 0

    def test_kelly_sizer_zero_cash(self):
        sizer = KellyCriterionSizer(fraction=0.5)
        row = pd.Series({"Close": 100.0, "kelly_win_rate": 0.6, "kelly_payoff_ratio": 2.0})
        shares = sizer.compute("TEST", 100.0, row, 100_000, 0, 0.10)
        assert shares == 0

    def test_risk_parity_sizer_zero_cash(self):
        sizer = RiskParitySizer(target_vol=0.10)
        row = pd.Series({"Close": 100.0, "ATR": 2.0})
        shares = sizer.compute("TEST", 100.0, row, 100_000, 0, 0.10)
        assert shares == 0

    def test_volatility_parity_zero_cash(self):
        sizer = VolatilityParity(target_vol=0.10)
        row = pd.Series({"Close": 100.0, "ATR": 2.0})
        shares = sizer.compute("TEST", 100.0, row, 100_000, 0, 0.10)
        assert shares == 0


class TestFixedFractionalMaxAllocAboveOne:
    """FixedFractional with max_alloc_pct > 1.0 (e.g., leveraged)."""

    def test_max_alloc_above_one(self):
        """When max_alloc_pct=2.0, the sizer should allocate up to 200% of equity,
        capped by available cash.
        """
        sizer = FixedFractional()
        row = pd.Series({"Close": 100.0})
        # equity=100k, max_alloc=2.0 => target=200k, but cash=150k => capped to 150k
        shares = sizer.compute("TEST", 100.0, row, 100_000, 150_000, 2.0)
        assert shares == 1500  # 150k / 100 = 1500

    def test_max_alloc_above_one_cash_limited(self):
        sizer = FixedFractional()
        row = pd.Series({"Close": 100.0})
        # equity=100k, max_alloc=1.5 => target=150k, but cash=50k => capped to 50k
        shares = sizer.compute("TEST", 100.0, row, 100_000, 50_000, 1.5)
        assert shares == 500


class TestKellyBreakEven:
    """Kelly criterion where f* = 0 (break-even edge)."""

    def test_kelly_breakeven_returns_zero(self):
        """When win_rate * payoff = (1 - win_rate), f* = 0 => 0 shares.

        Example: win_rate=0.5, payoff=1.0 => f* = (0.5*1 - 0.5)/1 = 0
        """
        sizer = KellyCriterionSizer(fraction=0.5)
        row = pd.Series({"Close": 100.0, "kelly_win_rate": 0.5, "kelly_payoff_ratio": 1.0})
        shares = sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        assert shares == 0

    def test_kelly_barely_positive(self):
        """Just above break-even: win_rate=0.51, payoff=1.0 => f*=0.02, half=0.01."""
        sizer = KellyCriterionSizer(fraction=0.5)
        row = pd.Series({"Close": 100.0, "kelly_win_rate": 0.51, "kelly_payoff_ratio": 1.0})
        shares = sizer.compute("TEST", 100.0, row, 100_000, 100_000, 0.10)
        # f* = (0.51*1 - 0.49)/1 = 0.02, half-kelly = 0.01
        # target = min(100k * 0.01, 100k * 0.10, 100k) = 1000 => 10 shares
        assert shares == 10
