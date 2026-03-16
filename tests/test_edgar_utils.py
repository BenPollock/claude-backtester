"""Tests for EDGAR utility functions (retry wrapper, rate limit detection)."""

import time
from unittest.mock import patch

import pandas as pd
import pytest

from backtester.data.edgar_utils import (
    DEFAULT_INITIAL_BACKOFF,
    DEFAULT_MAX_RETRIES,
    _is_rate_limit_error,
    edgar_retry,
)


# ---------------------------------------------------------------------------
# _is_rate_limit_error detection tests
# ---------------------------------------------------------------------------


class TestIsRateLimitError:
    """Test rate-limit error detection across various exception types."""

    def test_http_403_in_message(self):
        assert _is_rate_limit_error(Exception("HTTP 403 Forbidden")) is True

    def test_http_429_in_message(self):
        assert _is_rate_limit_error(Exception("HTTP 429 Too Many Requests")) is True

    def test_too_many_requests_text(self):
        assert _is_rate_limit_error(Exception("too many requests")) is True

    def test_rate_limit_text(self):
        assert _is_rate_limit_error(Exception("rate limit exceeded")) is True

    def test_rate_limit_case_insensitive(self):
        assert _is_rate_limit_error(Exception("Rate Limit Exceeded")) is True

    def test_class_name_too_many_requests(self):
        class TooManyRequestsError(Exception):
            pass
        assert _is_rate_limit_error(TooManyRequestsError("blocked")) is True

    def test_class_name_rate_limit(self):
        class RateLimitError(Exception):
            pass
        assert _is_rate_limit_error(RateLimitError("slow down")) is True

    def test_generic_error_not_rate_limit(self):
        assert _is_rate_limit_error(Exception("Connection timeout")) is False

    def test_value_error_not_rate_limit(self):
        assert _is_rate_limit_error(ValueError("invalid data")) is False

    def test_network_error_not_rate_limit(self):
        assert _is_rate_limit_error(ConnectionError("network unreachable")) is False

    def test_403_embedded_in_url_not_false_positive(self):
        # "403" in a URL path should still trigger (conservative approach)
        assert _is_rate_limit_error(Exception("GET /api returned 403")) is True

    def test_empty_message(self):
        assert _is_rate_limit_error(Exception("")) is False

    def test_timeout_not_rate_limit(self):
        """Timeout errors should not be treated as rate limits."""
        assert _is_rate_limit_error(TimeoutError("Request timed out")) is False

    def test_file_not_found_not_rate_limit(self):
        """FileNotFoundError is not a rate limit."""
        assert _is_rate_limit_error(FileNotFoundError("/path/missing")) is False

    def test_sec_specific_403_message(self):
        """SEC often returns plain '403' status text."""
        assert _is_rate_limit_error(Exception("Request returned status 403")) is True

    def test_mixed_case_class_name(self):
        """Class name detection is case-insensitive."""
        class tooManyRequestsError(Exception):
            pass
        assert _is_rate_limit_error(tooManyRequestsError("nope")) is True


# ---------------------------------------------------------------------------
# edgar_retry decorator tests
# ---------------------------------------------------------------------------


class TestEdgarRetry:
    """Test the retry decorator with simulated rate-limit errors."""

    def test_no_error_returns_immediately(self):
        """Function with no error should return its value on first call."""
        call_count = [0]

        @edgar_retry(max_retries=3, initial_backoff=0.01)
        def success():
            call_count[0] += 1
            return "ok"

        result = success()
        assert result == "ok"
        assert call_count[0] == 1

    def test_non_rate_limit_error_raises_immediately(self):
        """Non-rate-limit errors should raise without retry."""
        call_count = [0]

        @edgar_retry(max_retries=3, initial_backoff=0.01)
        def fails():
            call_count[0] += 1
            raise ValueError("bad data")

        with pytest.raises(ValueError, match="bad data"):
            fails()
        assert call_count[0] == 1  # no retries

    def test_rate_limit_error_retries(self):
        """Rate-limit error should trigger retries."""
        call_count = [0]

        @edgar_retry(max_retries=2, initial_backoff=0.01)
        def rate_limited_then_ok():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise Exception("HTTP 429 Too Many Requests")
            return "recovered"

        result = rate_limited_then_ok()
        assert result == "recovered"
        assert call_count[0] == 3  # 2 failures + 1 success

    def test_max_retries_exhausted(self):
        """After exhausting retries, the last exception is raised."""
        call_count = [0]

        @edgar_retry(max_retries=2, initial_backoff=0.01)
        def always_rate_limited():
            call_count[0] += 1
            raise Exception("HTTP 403 Forbidden")

        with pytest.raises(Exception, match="403"):
            always_rate_limited()
        assert call_count[0] == 3  # initial + 2 retries

    def test_exponential_backoff_timing(self):
        """Verify backoff delays increase exponentially."""
        call_count = [0]
        sleep_calls = []

        @edgar_retry(max_retries=2, initial_backoff=1.0)
        def always_fails():
            call_count[0] += 1
            raise Exception("HTTP 429")

        with patch("backtester.data.edgar_utils.time.sleep") as mock_sleep:
            with pytest.raises(Exception):
                always_fails()
            sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]

        # Should have 2 sleep calls: 1.0s and 2.0s
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == pytest.approx(1.0)
        assert sleep_calls[1] == pytest.approx(2.0)

    def test_403_triggers_retry(self):
        """HTTP 403 (SEC's primary response) triggers retry."""
        call_count = [0]

        @edgar_retry(max_retries=1, initial_backoff=0.01)
        def forbidden_then_ok():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("403 Forbidden")
            return "ok"

        result = forbidden_then_ok()
        assert result == "ok"
        assert call_count[0] == 2

    def test_custom_too_many_requests_class(self):
        """edgartools-style TooManyRequestsError triggers retry."""
        call_count = [0]

        class TooManyRequestsError(Exception):
            pass

        @edgar_retry(max_retries=1, initial_backoff=0.01)
        def edgartools_rate_limited():
            call_count[0] += 1
            if call_count[0] == 1:
                raise TooManyRequestsError("slow down")
            return "ok"

        result = edgartools_rate_limited()
        assert result == "ok"

    def test_zero_retries(self):
        """With max_retries=0, no retries happen."""
        call_count = [0]

        @edgar_retry(max_retries=0, initial_backoff=0.01)
        def fails():
            call_count[0] += 1
            raise Exception("HTTP 429")

        with pytest.raises(Exception, match="429"):
            fails()
        assert call_count[0] == 1

    def test_preserves_function_metadata(self):
        """Decorator should preserve the wrapped function's name and docstring."""

        @edgar_retry()
        def my_fetch():
            """Fetch data from EDGAR."""
            pass

        assert my_fetch.__name__ == "my_fetch"
        assert my_fetch.__doc__ == "Fetch data from EDGAR."

    def test_default_parameters(self):
        """Verify default retry parameters are sensible."""
        assert DEFAULT_MAX_RETRIES == 3
        assert DEFAULT_INITIAL_BACKOFF == 10.0

    @patch("backtester.data.edgar_utils.time.sleep")
    def test_three_retries_backoff_values(self, mock_sleep):
        """With 3 retries and initial_backoff=10, sleeps are 10, 20, 40."""
        call_count = [0]

        @edgar_retry(max_retries=3, initial_backoff=10.0)
        def always_limited():
            call_count[0] += 1
            raise Exception("HTTP 403")

        with pytest.raises(Exception):
            always_limited()

        assert call_count[0] == 4  # 1 initial + 3 retries
        assert mock_sleep.call_count == 3
        calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert calls == [10.0, 20.0, 40.0]

    @patch("backtester.data.edgar_utils.time.sleep")
    def test_retry_preserves_args_and_kwargs(self, mock_sleep):
        """Arguments and keyword arguments are passed through on retry."""
        call_count = [0]

        @edgar_retry(max_retries=1, initial_backoff=0.01)
        def fetch(symbol, form="4"):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("429 Too Many Requests")
            return f"{symbol}:{form}"

        result = fetch("AAPL", form="10-K")
        assert result == "AAPL:10-K"
        assert call_count[0] == 2

    @patch("backtester.data.edgar_utils.time.sleep")
    def test_retry_on_rate_limit_text_error(self, mock_sleep):
        """Errors containing 'rate limit' text trigger retry."""
        call_count = [0]

        @edgar_retry(max_retries=1, initial_backoff=0.01)
        def func():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("SEC EDGAR rate limit exceeded, try again later")
            return "ok"

        assert func() == "ok"
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# Integration: retry with DataFrame-returning functions
# ---------------------------------------------------------------------------


class TestEdgarRetryIntegration:
    """Test retry decorator with DataFrame-returning functions (like EDGAR sources)."""

    def test_retry_returns_dataframe(self):
        """After recovery, a DataFrame return value is preserved."""
        call_count = [0]

        @edgar_retry(max_retries=1, initial_backoff=0.01)
        def fetch_data():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("HTTP 429")
            return pd.DataFrame({"metric": ["revenue"], "value": [1e9]})

        result = fetch_data()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["value"].iloc[0] == 1e9

    @patch("backtester.data.edgar_utils.time.sleep")
    def test_source_class_method_retry(self, mock_sleep):
        """Retry works on class methods (like EdgarInsiderSource.fetch)."""
        call_count = [0]

        class FakeEdgarSource:
            @edgar_retry(max_retries=2, initial_backoff=0.01)
            def fetch(self, symbol):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise Exception("HTTP 403 Forbidden")
                return pd.DataFrame({"filed_date": ["2020-01-01"], "shares": [1000]})

        source = FakeEdgarSource()
        result = source.fetch("AAPL")
        assert len(result) == 1
        assert call_count[0] == 2

    @patch("backtester.data.edgar_utils.time.sleep")
    def test_non_rate_limit_error_not_retried_in_source(self, mock_sleep):
        """TypeError in a source fetch is not retried."""
        call_count = [0]

        class BadSource:
            @edgar_retry(max_retries=3, initial_backoff=0.01)
            def fetch(self, symbol):
                call_count[0] += 1
                raise TypeError("unexpected None value")

        source = BadSource()
        with pytest.raises(TypeError, match="unexpected None"):
            source.fetch("AAPL")
        assert call_count[0] == 1
        mock_sleep.assert_not_called()

    @patch("backtester.data.edgar_utils.time.sleep")
    def test_intermittent_failure_recovers(self, mock_sleep):
        """Source that fails twice then succeeds on third attempt."""
        call_count = [0]

        class IntermittentSource:
            @edgar_retry(max_retries=3, initial_backoff=0.01)
            def fetch(self, symbol):
                call_count[0] += 1
                if call_count[0] <= 2:
                    raise Exception("HTTP 429 Too Many Requests")
                return pd.DataFrame({"metric": ["eps"], "value": [2.5]})

        source = IntermittentSource()
        result = source.fetch("MSFT")
        assert len(result) == 1
        assert result["value"].iloc[0] == 2.5
        assert call_count[0] == 3
        assert mock_sleep.call_count == 2
