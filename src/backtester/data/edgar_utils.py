"""Shared utilities for SEC EDGAR data sources.

Provides a retry wrapper for EDGAR API calls that handles rate limiting
(HTTP 403/429) with exponential backoff. The edgartools library already
has built-in rate limiting (9 req/sec via leaky bucket), so we don't
implement our own rate limiter — this wrapper only handles transient
failures when the SEC temporarily blocks our IP.
"""

import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

# Default retry settings — SEC bans can last up to 10 minutes,
# so we use long backoff intervals.
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_BACKOFF = 10.0  # seconds


def edgar_retry(max_retries=DEFAULT_MAX_RETRIES, initial_backoff=DEFAULT_INITIAL_BACKOFF):
    """Decorator that retries on SEC rate-limit errors (HTTP 403/429).

    Uses exponential backoff: 10s, 20s, 40s by default.

    Catches:
    - HTTP 403 (SEC's primary rate-limit response)
    - HTTP 429 (Too Many Requests)
    - TooManyRequestsError from edgartools (if available)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if not _is_rate_limit_error(exc):
                        raise
                    last_exc = exc
                    if attempt < max_retries:
                        delay = initial_backoff * (2 ** attempt)
                        logger.warning(
                            "SEC rate limit hit (attempt %d/%d), "
                            "retrying in %.0fs: %s",
                            attempt + 1, max_retries + 1, delay, exc,
                        )
                        time.sleep(delay)
                    else:
                        logger.warning(
                            "SEC rate limit: max retries (%d) exhausted: %s",
                            max_retries + 1, exc,
                        )
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a SEC rate-limit error."""
    exc_str = str(exc).lower()

    # Check for HTTP status codes in error message
    if "403" in exc_str or "429" in exc_str:
        return True

    # Check for "too many requests" text
    if "too many requests" in exc_str:
        return True

    # Check for "rate limit" text
    if "rate limit" in exc_str:
        return True

    # Check exception class name (edgartools may raise TooManyRequestsError)
    cls_name = type(exc).__name__.lower()
    if "toomanyrequest" in cls_name or "ratelimit" in cls_name:
        return True

    return False
