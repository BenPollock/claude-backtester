"""Core enumerations used across the backtester."""

from enum import Enum, auto


class Side(Enum):
    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()       # triggers market sell when price <= stop_price
    STOP_LIMIT = auto() # triggers limit sell when price <= stop_price


class OrderStatus(Enum):
    PENDING = auto()
    FILLED = auto()
    CANCELLED = auto()


class SignalAction(Enum):
    BUY = auto()
    SELL = auto()
    HOLD = auto()
    SHORT = auto()
    COVER = auto()
