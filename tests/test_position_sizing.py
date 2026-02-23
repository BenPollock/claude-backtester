"""Tests for position sizing models."""

import pandas as pd
import pytest

from backtester.execution.position_sizing import (
    FixedFractional,
    ATRSizer,
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
