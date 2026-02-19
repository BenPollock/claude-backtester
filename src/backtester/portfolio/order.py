"""Order, Fill, and Trade dataclasses."""

from dataclasses import dataclass, field
from datetime import date
import uuid

from backtester.types import Side, OrderType, OrderStatus


@dataclass
class Order:
    symbol: str
    side: Side
    quantity: int
    order_type: OrderType
    signal_date: date  # close of this day generated the signal
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    reason: str = ""  # e.g. "overallocation"


@dataclass(frozen=True)
class Fill:
    order_id: str
    symbol: str
    side: Side
    quantity: int
    price: float  # after slippage
    commission: float
    fill_date: date  # T+1
    slippage: float  # slippage amount in price terms


@dataclass(frozen=True)
class TradeLogEntry:
    """Per-fill activity log entry for every BUY and SELL."""

    date: date
    symbol: str
    action: Side          # BUY or SELL
    quantity: int
    price: float          # fill price (after slippage)
    value: float          # quantity * price
    avg_cost_basis: float | None  # SELL only: weighted avg entry price before sale
    fees: float           # commission
    slippage: float       # slippage in price terms


@dataclass
class Trade:
    """Completed round-trip trade."""

    symbol: str
    entry_date: date
    exit_date: date
    entry_price: float  # avg across lots
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    holding_days: int
    fees_total: float
