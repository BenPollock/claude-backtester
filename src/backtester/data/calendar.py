"""NYSE trading calendar wrapper using exchange_calendars."""

import logging
from datetime import date

import pandas as pd
import exchange_calendars as xcals

logger = logging.getLogger(__name__)


class TradingCalendar:
    """Provides NYSE trading days for date range alignment.

    Falls back to pandas business days for dates outside the
    exchange_calendars supported range.
    """

    def __init__(self, exchange: str = "XNYS"):
        self._calendar = xcals.get_calendar(exchange)
        self._cal_start = self._calendar.first_session.date()
        self._cal_end = self._calendar.last_session.date()

    def trading_days(self, start: date, end: date) -> pd.DatetimeIndex:
        """Return sorted DatetimeIndex of trading days in [start, end]."""
        # Clamp to calendar's supported range
        clamped_start = max(start, self._cal_start)
        clamped_end = min(end, self._cal_end)

        if clamped_start <= clamped_end:
            sessions = self._calendar.sessions_in_range(
                pd.Timestamp(clamped_start), pd.Timestamp(clamped_end)
            )
            cal_days = pd.DatetimeIndex(sessions.date, name="Date")
        else:
            cal_days = pd.DatetimeIndex([], name="Date")

        # Fill dates outside the calendar range with business days
        parts = []
        if start < self._cal_start:
            logger.debug(f"Calendar starts at {self._cal_start}; using business days for {start} to {self._cal_start}")
            pre = pd.bdate_range(start=start, end=self._cal_start - pd.Timedelta(days=1), freq="B")
            parts.append(pd.DatetimeIndex(pre.date, name="Date"))

        parts.append(cal_days)

        if end > self._cal_end:
            logger.debug(f"Calendar ends at {self._cal_end}; using business days for {self._cal_end} to {end}")
            post = pd.bdate_range(start=self._cal_end + pd.Timedelta(days=1), end=end, freq="B")
            parts.append(pd.DatetimeIndex(post.date, name="Date"))

        combined = parts[0]
        for p in parts[1:]:
            combined = combined.append(p)

        return combined.unique().sort_values()

    def is_trading_day(self, d: date) -> bool:
        ts = pd.Timestamp(d)
        if ts < self._calendar.first_session or ts > self._calendar.last_session:
            # Fallback: weekday check
            return ts.weekday() < 5
        return self._calendar.is_session(ts)

    def next_trading_day(self, d: date) -> date:
        """Return next trading day after d (exclusive)."""
        ts = pd.Timestamp(d)
        if ts < self._calendar.last_session:
            try:
                return self._calendar.next_session(ts).date()
            except ValueError:
                pass
        # Fallback: next weekday
        nxt = ts + pd.Timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += pd.Timedelta(days=1)
        return nxt.date()
