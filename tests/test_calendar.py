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


class TestNYSEHolidays:
    """Verify that NYSE holidays are excluded from trading days."""

    def setup_method(self):
        self.cal = TradingCalendar()

    def test_holidays_excluded_from_trading_days_range(self):
        """trading_days() for a week containing MLK Day should skip it."""
        cal = self.cal
        # Week of MLK Day 2024: Jan 15 is the holiday
        days = cal.trading_days(date(2024, 1, 15), date(2024, 1, 19))
        day_dates = [d.date() if hasattr(d, "date") else d for d in days]
        assert date(2024, 1, 15) not in day_dates
        # But Tuesday through Friday should be present
        assert date(2024, 1, 16) in day_dates
        assert date(2024, 1, 17) in day_dates
        assert date(2024, 1, 18) in day_dates
        assert date(2024, 1, 19) in day_dates


class TestTradingDaysEdgeCases:
    """Edge cases for trading_days()."""

    def setup_method(self):
        self.cal = TradingCalendar()

    def test_start_after_end_returns_empty(self):
        days = self.cal.trading_days(date(2024, 6, 15), date(2024, 6, 10))
        assert len(days) == 0

    def test_start_equals_end_on_trading_day(self):
        # 2024-01-09 is a Tuesday (trading day)
        days = self.cal.trading_days(date(2024, 1, 9), date(2024, 1, 9))
        assert len(days) == 1
        dt = days[0].date() if hasattr(days[0], "date") else days[0]
        assert dt == date(2024, 1, 9)

    def test_start_equals_end_on_weekend(self):
        # 2024-01-13 is a Saturday — no trading days
        days = self.cal.trading_days(date(2024, 1, 13), date(2024, 1, 13))
        assert len(days) == 0

    def test_start_equals_end_on_holiday(self):
        # Christmas 2024 is a Wednesday — holiday, no trading
        days = self.cal.trading_days(date(2024, 12, 25), date(2024, 12, 25))
        assert len(days) == 0

    def test_year_boundary_dec31_to_jan2(self):
        """Dec 31, 2023 (Sun) to Jan 2, 2024 (Tue): only Jan 2 is a trading day.
        Jan 1 is New Year's Day (holiday)."""
        days = self.cal.trading_days(date(2023, 12, 31), date(2024, 1, 2))
        day_dates = [d.date() if hasattr(d, "date") else d for d in days]
        assert date(2024, 1, 1) not in day_dates  # holiday
        assert date(2024, 1, 2) in day_dates  # first trading day of 2024

    def test_year_boundary_dec_to_jan_count(self):
        """Full week spanning year boundary 2023-2024."""
        days = self.cal.trading_days(date(2023, 12, 28), date(2024, 1, 5))
        day_dates = [d.date() if hasattr(d, "date") else d for d in days]
        # Dec 28 (Thu), Dec 29 (Fri) are trading days
        assert date(2023, 12, 28) in day_dates
        assert date(2023, 12, 29) in day_dates
        # Dec 30 (Sat), Dec 31 (Sun), Jan 1 (holiday) are NOT trading days
        assert date(2023, 12, 30) not in day_dates
        assert date(2023, 12, 31) not in day_dates
        assert date(2024, 1, 1) not in day_dates
        # Jan 2 (Tue) through Jan 5 (Fri) are trading days
        assert date(2024, 1, 2) in day_dates
        assert date(2024, 1, 3) in day_dates

    def test_trading_days_sorted(self):
        """Returned trading days should always be sorted ascending."""
        days = self.cal.trading_days(date(2024, 1, 1), date(2024, 3, 31))
        for i in range(1, len(days)):
            assert days[i] > days[i - 1]


class TestNextTradingDayHolidays:
    """next_trading_day() should skip holidays."""

    def setup_method(self):
        self.cal = TradingCalendar()

    def test_friday_to_monday(self):
        # 2024-01-12 (Fri) → 2024-01-16 (Tue) because Jan 15 is MLK Day
        nxt = self.cal.next_trading_day(date(2024, 1, 12))
        assert nxt == date(2024, 1, 16)

    def test_before_independence_day(self):
        # 2024-07-03 (Wed) → 2024-07-05 (Fri) because July 4 is a holiday
        nxt = self.cal.next_trading_day(date(2024, 7, 3))
        assert nxt == date(2024, 7, 5)

    def test_before_christmas(self):
        # 2024-12-24 (Tue) → 2024-12-26 (Thu) because Dec 25 is Christmas
        nxt = self.cal.next_trading_day(date(2024, 12, 24))
        assert nxt == date(2024, 12, 26)

    def test_new_years_eve_to_jan2(self):
        # 2023-12-29 (Fri) → 2024-01-02 (Tue)
        # Skips weekend (Dec 30-31) and New Year's Day (Jan 1)
        nxt = self.cal.next_trading_day(date(2023, 12, 29))
        assert nxt == date(2024, 1, 2)

    def test_wednesday_to_thursday_no_holiday(self):
        # 2024-01-10 (Wed) → 2024-01-11 (Thu), no holidays
        nxt = self.cal.next_trading_day(date(2024, 1, 10))
        assert nxt == date(2024, 1, 11)


class TestIsTradingDayHolidays:
    """is_trading_day() for specific known holidays."""

    def setup_method(self):
        self.cal = TradingCalendar()

    def test_day_after_holiday_is_trading_day(self):
        # Jan 16, 2024 (Tue after MLK Day) — trading day
        assert self.cal.is_trading_day(date(2024, 1, 16)) is True

    def test_multiple_holidays_in_year(self):
        """Verify all major 2024 NYSE holidays are non-trading days."""
        holidays_2024 = [
            date(2024, 1, 1),   # New Year's
            date(2024, 1, 15),  # MLK Day
            date(2024, 2, 19),  # Presidents Day
            date(2024, 3, 29),  # Good Friday
            date(2024, 5, 27),  # Memorial Day
            date(2024, 6, 19),  # Juneteenth
            date(2024, 7, 4),   # Independence Day
            date(2024, 9, 2),   # Labor Day
            date(2024, 11, 28), # Thanksgiving
            date(2024, 12, 25), # Christmas
        ]
        for holiday in holidays_2024:
            assert self.cal.is_trading_day(holiday) is False, (
                f"{holiday} should be an NYSE holiday"
            )
