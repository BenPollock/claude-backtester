"""Tests for TradingCalendar."""

from datetime import date

from backtester.data.calendar import TradingCalendar


class TestTradingCalendar:
    def test_trading_days_excludes_weekends(self):
        cal = TradingCalendar()
        # 2024-01-08 (Mon) to 2024-01-19 (Fri) — 2 full weeks
        days = cal.trading_days(date(2024, 1, 8), date(2024, 1, 19))
        for d in days:
            dt = d.date() if hasattr(d, 'date') else d
            assert dt.weekday() < 5, f"{dt} is a weekend day"

    def test_is_trading_day_weekday_vs_weekend(self):
        cal = TradingCalendar()
        # 2024-01-09 is a Tuesday
        assert cal.is_trading_day(date(2024, 1, 9)) is True
        # 2024-01-13 is a Saturday
        assert cal.is_trading_day(date(2024, 1, 13)) is False

    def test_next_trading_day_skips_weekend(self):
        cal = TradingCalendar()
        # 2024-01-05 is a Friday (no holiday on Jan 8) → next = Monday 2024-01-08
        nxt = cal.next_trading_day(date(2024, 1, 5))
        assert nxt == date(2024, 1, 8)
