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
    """Holds lots for a single symbol. Sells use FIFO ordering.

    Supports both long (positive quantity) and short (negative quantity)
    positions.  Short positions are represented by lots with negative
    ``quantity`` values.  The ``direction`` property indicates the current
    side of the position.
    """

    symbol: str
    lots: list[Lot] = field(default_factory=list)
    _market_price: float = 0.0
    stop_state: StopState = field(default_factory=StopState)
    short_borrow_cost_accrued: float = 0.0

    # ---- direction helpers ----

    @property
    def direction(self) -> str:
        """Return 'long', 'short', or 'flat'."""
        qty = self.total_quantity
        if qty > 0:
            return "long"
        elif qty < 0:
            return "short"
        return "flat"

    @property
    def is_short(self) -> bool:
        return self.total_quantity < 0

    # ---- quantity / value ----

    @property
    def total_quantity(self) -> int:
        return sum(lot.quantity for lot in self.lots)

    @property
    def avg_entry_price(self) -> float:
        total_qty = self.total_quantity
        if total_qty == 0:
            return 0.0
        # For short positions, lots have negative qty; use abs for weighted avg
        abs_qty = abs(total_qty)
        return sum(lot.entry_price * abs(lot.quantity) for lot in self.lots) / abs_qty

    @property
    def market_value(self) -> float:
        """Market value of the position.

        For long positions this is positive (qty * price).
        For short positions this is negative (negative qty * price).
        """
        return self.total_quantity * self._market_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized PnL for the position (excluding borrow costs).

        Long:  (market_price - avg_entry) * quantity
        Short: (avg_entry - market_price) * abs(quantity)
        """
        qty = self.total_quantity
        if qty == 0 or self._market_price == 0:
            return 0.0
        if qty > 0:
            return (self._market_price - self.avg_entry_price) * qty
        else:
            return (self.avg_entry_price - self._market_price) * abs(qty)

    def update_market_price(self, price: float) -> None:
        self._market_price = price

    def add_lot(self, quantity: int, price: float, entry_date: date, commission: float = 0.0) -> None:
        self.lots.append(Lot(quantity=quantity, entry_price=price, entry_date=entry_date, commission=commission))

    def accrue_borrow_cost(self, rate_annual: float, days: int = 1) -> float:
        """Accrue short borrow cost and return the daily cost.

        cost = abs(market_value) * rate_annual / 252 * days
        Only meaningful for short positions. Returns the cost deducted.
        """
        if self.total_quantity >= 0:
            return 0.0
        daily_cost = abs(self.market_value) * rate_annual / 252 * days
        self.short_borrow_cost_accrued += daily_cost
        return daily_cost

    # ---- FIFO close logic ----

    def sell_lots_fifo(self, quantity: int, exit_price: float, exit_date: date, exit_commission: float = 0.0) -> list[Trade]:
        """Sell ``quantity`` shares of a long position using FIFO.

        Returns completed Trade records.  This is the original long-only
        method, preserved for backward compatibility.
        """
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

    def sell_lots_lifo(self, quantity: int, exit_price: float, exit_date: date, exit_commission: float = 0.0) -> list[Trade]:
        """Sell ``quantity`` shares of a long position using LIFO (Gap 36)."""
        if quantity > self.total_quantity:
            raise ValueError(f"Cannot sell {quantity} shares of {self.symbol}; only hold {self.total_quantity}")

        trades: list[Trade] = []
        remaining = quantity
        commission_per_share = exit_commission / quantity if quantity > 0 else 0.0

        while remaining > 0 and self.lots:
            lot = self.lots[-1]  # LIFO: take from end
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
                self.lots.pop()  # remove from end
            else:
                lot.commission -= entry_comm

            remaining -= sell_qty

        return trades

    def sell_lots_by_cost(self, quantity: int, exit_price: float, exit_date: date,
                          exit_commission: float = 0.0, highest_first: bool = True) -> list[Trade]:
        """Sell ``quantity`` shares sorted by cost basis (Gap 36)."""
        if quantity > self.total_quantity:
            raise ValueError(f"Cannot sell {quantity} shares of {self.symbol}; only hold {self.total_quantity}")

        # Sort lots by entry_price
        self.lots.sort(key=lambda lot: lot.entry_price, reverse=highest_first)

        trades: list[Trade] = []
        remaining = quantity
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
                lot.commission -= entry_comm

            remaining -= sell_qty

        return trades

    def close_lots_fifo(self, quantity: int, exit_price: float, exit_date: date, exit_commission: float = 0.0) -> list[Trade]:
        """Close ``quantity`` shares/contracts of a short position using FIFO.

        ``quantity`` is expressed as a positive number of shares to cover.
        Lots have negative quantities; this method removes shares from the
        front of the lot list.

        PnL for shorts: (entry_price - exit_price) * qty - fees
        (profit when price drops).
        """
        abs_held = abs(self.total_quantity)
        if quantity > abs_held:
            raise ValueError(
                f"Cannot cover {quantity} shares of {self.symbol}; "
                f"only short {abs_held}"
            )

        trades: list[Trade] = []
        remaining = quantity
        commission_per_share = exit_commission / quantity if quantity > 0 else 0.0

        while remaining > 0 and self.lots:
            lot = self.lots[0]
            lot_abs = abs(lot.quantity)
            cover_qty = min(remaining, lot_abs)

            entry_comm = lot.commission * (cover_qty / lot_abs) if lot_abs > 0 else 0.0
            exit_comm = commission_per_share * cover_qty
            total_fees = entry_comm + exit_comm

            # Short PnL: profit when price falls
            pnl = (lot.entry_price - exit_price) * cover_qty - total_fees
            pnl_pct = (lot.entry_price / exit_price - 1.0) if exit_price > 0 else 0.0

            trades.append(Trade(
                symbol=self.symbol,
                entry_date=lot.entry_date,
                exit_date=exit_date,
                entry_price=lot.entry_price,
                exit_price=exit_price,
                quantity=cover_qty,
                pnl=pnl,
                pnl_pct=pnl_pct,
                holding_days=(exit_date - lot.entry_date).days,
                fees_total=total_fees,
            ))

            # Remove covered shares from the lot (lots are negative)
            lot.quantity += cover_qty  # e.g. -100 + 30 = -70
            if lot.quantity == 0:
                self.lots.pop(0)
            else:
                lot.commission -= entry_comm

            remaining -= cover_qty

        return trades
