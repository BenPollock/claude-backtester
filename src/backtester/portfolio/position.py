"""Position and Lot management with FIFO sell logic."""

from dataclasses import dataclass, field
from datetime import date

from backtester.portfolio.order import Trade


@dataclass
class Lot:
    quantity: int
    entry_price: float
    entry_date: date
    commission: float = 0.0  # entry commission allocated to this lot


@dataclass
class StopState:
    """Active stop-loss / take-profit / trailing-stop levels for a position."""

    stop_loss: float | None = None     # absolute price level
    take_profit: float | None = None   # absolute price level
    trailing_stop_pct: float | None = None  # e.g. 0.05 = 5% trail
    trailing_high: float = 0.0         # high-water mark for trailing stop

    @property
    def trailing_stop_price(self) -> float | None:
        """Current trailing stop trigger price."""
        if self.trailing_stop_pct is None or self.trailing_high <= 0:
            return None
        return self.trailing_high * (1.0 - self.trailing_stop_pct)


@dataclass
class Position:
    """Holds lots for a single symbol. Sells use FIFO ordering."""

    symbol: str
    lots: list[Lot] = field(default_factory=list)
    _market_price: float = 0.0
    stop_state: StopState = field(default_factory=StopState)

    @property
    def total_quantity(self) -> int:
        return sum(lot.quantity for lot in self.lots)

    @property
    def avg_entry_price(self) -> float:
        total_qty = self.total_quantity
        if total_qty == 0:
            return 0.0
        return sum(lot.entry_price * lot.quantity for lot in self.lots) / total_qty

    @property
    def market_value(self) -> float:
        return self.total_quantity * self._market_price

    def update_market_price(self, price: float) -> None:
        self._market_price = price

    def add_lot(self, quantity: int, price: float, entry_date: date, commission: float = 0.0) -> None:
        self.lots.append(Lot(quantity=quantity, entry_price=price, entry_date=entry_date, commission=commission))

    def sell_lots_fifo(self, quantity: int, exit_price: float, exit_date: date, exit_commission: float = 0.0) -> list[Trade]:
        """Sell `quantity` shares using FIFO. Returns completed Trade records."""
        if quantity > self.total_quantity:
            raise ValueError(f"Cannot sell {quantity} shares of {self.symbol}; only hold {self.total_quantity}")

        trades: list[Trade] = []
        remaining = quantity
        # Allocate exit commission proportionally
        commission_per_share = exit_commission / quantity if quantity > 0 else 0.0

        while remaining > 0 and self.lots:
            lot = self.lots[0]
            sell_qty = min(remaining, lot.quantity)

            entry_comm = lot.commission * (sell_qty / lot.quantity) if lot.quantity > 0 else 0.0
            exit_comm = commission_per_share * sell_qty
            total_fees = entry_comm + exit_comm

            pnl = (exit_price - lot.entry_price) * sell_qty - total_fees
            pnl_pct = (exit_price / lot.entry_price - 1.0) if lot.entry_price > 0 else 0.0

            trades.append(Trade(
                symbol=self.symbol,
                entry_date=lot.entry_date,
                exit_date=exit_date,
                entry_price=lot.entry_price,
                exit_price=exit_price,
                quantity=sell_qty,
                pnl=pnl,
                pnl_pct=pnl_pct,
                holding_days=(exit_date - lot.entry_date).days,
                fees_total=total_fees,
            ))

            lot.quantity -= sell_qty
            if lot.quantity == 0:
                self.lots.pop(0)
            else:
                # Adjust remaining commission for partial lot
                lot.commission -= entry_comm

            remaining -= sell_qty

        return trades
