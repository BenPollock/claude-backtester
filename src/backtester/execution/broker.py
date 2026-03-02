"""Simulated broker: manages pending orders and fills them at market open."""

import logging
import random
from datetime import date

import pandas as pd

from backtester.portfolio.order import Order, Fill, TradeLogEntry
from backtester.portfolio.portfolio import Portfolio
from backtester.execution.slippage import SlippageModel, FixedSlippage
from backtester.execution.fees import FeeModel, PerTradeFee
from backtester.types import Side, OrderType, OrderStatus

logger = logging.getLogger(__name__)


class SimulatedBroker:
    """Accepts orders and fills them at the next day's open price + slippage + fees.

    Limit orders fill at the limit price (+ slippage) when the day's
    price range reaches the limit.  DAY orders expire at the end of the
    fill attempt; GTC orders persist across days until filled or expired.
    """

    def __init__(self, slippage: SlippageModel | None = None, fees: FeeModel | None = None,
                 max_volume_pct: float = 0.10, partial_fill_policy: str = "cancel",
                 fill_price_model: str = "open"):
        self._slippage = slippage or FixedSlippage()
        self._fees = fees or PerTradeFee()
        self._pending_orders: list[Order] = []
        self._max_volume_pct = max_volume_pct
        self._partial_fill_policy = partial_fill_policy
        self._fill_price_model = fill_price_model
        # Gap 13: Bracket order tracking
        self._bracket_children: dict[str, list[Order]] = {}  # parent_id -> child orders
        self._active_brackets: dict[str, list[Order]] = {}  # parent_id -> active children

    def submit_order(self, order: Order) -> None:
        """Queue an order for fill on the next trading day."""
        self._pending_orders.append(order)
        logger.debug(f"Order submitted: {order.side.name} {order.quantity} {order.symbol}")

    def submit_bracket(self, entry: Order, stop_loss: Order, take_profit: Order) -> None:
        """Submit a bracket order: entry + two OCO children (Gap 13).

        When the entry fills, both children become active. When either
        child fills, the other is cancelled.
        """
        stop_loss.parent_id = entry.id
        take_profit.parent_id = entry.id
        self._bracket_children[entry.id] = [stop_loss, take_profit]
        self._pending_orders.append(entry)
        logger.debug(f"Bracket submitted: entry={entry.symbol} {entry.side.name}")

    @property
    def pending_orders(self) -> list[Order]:
        return list(self._pending_orders)

    def _determine_fill_price(
        self, order: Order, row: pd.Series
    ) -> float | None:
        """Return the base fill price for the order, or None if unfillable.

        Market orders fill based on fill_price_model. Limit orders fill at
        the limit price if the day's range reaches it. Stop orders trigger
        when price crosses the stop level.
        """
        open_price = row["Open"]
        low = row.get("Low", open_price)
        high = row.get("High", open_price)
        close = row.get("Close", open_price)

        # Gap 11: STOP orders
        if order.order_type == OrderType.STOP and order.stop_price is not None:
            if order.side == Side.SELL:
                # STOP SELL: triggers when Low <= stop_price
                if low <= order.stop_price:
                    return order.stop_price
            else:
                # STOP BUY: triggers when High >= stop_price
                if high >= order.stop_price:
                    return order.stop_price
            return None  # not triggered

        if order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None:
                return None
            # Check if stop is triggered first
            triggered = False
            if order.side == Side.SELL and low <= order.stop_price:
                triggered = True
            elif order.side == Side.BUY and high >= order.stop_price:
                triggered = True
            if not triggered:
                return None
            # Stop triggered — now check limit
            if order.limit_price is not None:
                if order.side == Side.SELL and high >= order.limit_price:
                    return order.limit_price
                elif order.side == Side.BUY and low <= order.limit_price:
                    return order.limit_price
            return None  # stop triggered but limit not reachable

        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            limit = order.limit_price
            if order.side == Side.BUY:
                if low <= limit:
                    return limit
            else:
                if high >= limit:
                    return limit
            return None  # limit not reached

        # Market order — use fill_price_model (Gap 14)
        if self._fill_price_model == "close":
            return close
        elif self._fill_price_model == "vwap":
            return (high + low + close) / 3.0
        elif self._fill_price_model == "random":
            return random.uniform(low, high)
        else:
            # Default: open
            return open_price

    def process_fills(
        self,
        current_date: date,
        market_data: dict[str, pd.Series],
        portfolio: Portfolio,
    ) -> list[Fill]:
        """Fill pending orders using today's OHLCV data.

        Market orders fill at today's Open + slippage.  Limit orders fill
        at the limit price + slippage when the day's High/Low range
        reaches the limit.  DAY orders that cannot fill are cancelled;
        GTC orders are kept pending (with ``days_pending`` incremented)
        until filled or expired.

        Args:
            current_date: Today's date (fill date)
            market_data: dict of symbol -> today's OHLCV row (pd.Series)
            portfolio: Portfolio to update with fills

        Returns:
            List of Fill objects for orders filled today
        """
        fills: list[Fill] = []
        remaining_orders: list[Order] = []

        for order in self._pending_orders:
            row = market_data.get(order.symbol)
            if row is None or pd.isna(row.get("Open")):
                # No market data -- keep order pending
                remaining_orders.append(order)
                continue

            # Check if the order can fill at this price level
            base_price = self._determine_fill_price(order, row)
            if base_price is None:
                # Limit/stop price not reached today
                self._handle_unfilled_order(order, current_date, remaining_orders)
                continue

            open_price = row["Open"]
            volume = row.get("Volume", 0)

            # For SELL orders, resolve quantity if sentinel (-1 = sell all)
            quantity = order.quantity
            if order.side == Side.SELL and quantity < 0:
                pos = portfolio.get_position(order.symbol)
                if pos is None or pos.total_quantity == 0:
                    order.status = OrderStatus.CANCELLED
                    continue
                # For short positions being covered, resolve to abs of held qty
                if pos.is_short:
                    quantity = abs(pos.total_quantity)
                else:
                    quantity = pos.total_quantity

            # For BUY orders that are covers (quantity < 0 sentinel), resolve
            if order.side == Side.BUY and quantity < 0:
                pos = portfolio.get_position(order.symbol)
                if pos is None or not pos.is_short:
                    order.status = OrderStatus.CANCELLED
                    continue
                quantity = abs(pos.total_quantity)

            # Gap 3: Volume-constrained partial fills
            if volume > 0 and self._max_volume_pct > 0:
                max_fillable = int(volume * self._max_volume_pct)
                if max_fillable > 0 and quantity > max_fillable:
                    if self._partial_fill_policy == "requeue":
                        remainder_qty = quantity - max_fillable
                        remainder_order = Order(
                            symbol=order.symbol,
                            side=order.side,
                            quantity=remainder_qty,
                            order_type=order.order_type,
                            signal_date=order.signal_date,
                            limit_price=order.limit_price,
                            stop_price=order.stop_price,
                            time_in_force="GTC",
                            reason=order.reason,
                        )
                        remaining_orders.append(remainder_order)
                    quantity = max_fillable

            # Apply slippage around the base fill price
            fill_price = self._slippage.compute(order, base_price, volume)
            commission = self._fees.compute(order, fill_price, abs(quantity))

            if order.side == Side.BUY:
                total_cost = fill_price * quantity + commission
                if total_cost > portfolio.cash:
                    # Reduce quantity to fit budget
                    quantity = int((portfolio.cash - commission) // fill_price)
                    if quantity <= 0:
                        order.status = OrderStatus.CANCELLED
                        logger.debug(f"Cancelled {order.symbol} BUY: insufficient cash")
                        continue
                    commission = self._fees.compute(order, fill_price, quantity)

            elif order.side == Side.SELL and quantity < 0:
                # This is a short entry (negative quantity on a SELL order)
                quantity = abs(quantity)

            # Execute fill -- slippage is relative to the base fill price
            slippage_amount = abs(fill_price - base_price)
            fill = Fill(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                quantity=quantity,
                price=fill_price,
                commission=commission,
                fill_date=current_date,
                slippage=slippage_amount,
            )

            # Determine if this is a short-related order via the reason field
            is_short_entry = order.reason == "short_entry"
            is_cover = order.reason == "cover"

            # Update portfolio
            if order.side == Side.BUY and not is_cover:
                # Normal long buy
                pos = portfolio.open_position(order.symbol)
                pos.add_lot(quantity, fill_price, current_date, commission)
                portfolio.cash -= fill_price * quantity + commission
                portfolio.activity_log.append(TradeLogEntry(
                    date=current_date, symbol=order.symbol, action=Side.BUY,
                    quantity=quantity, price=fill_price,
                    value=quantity * fill_price, avg_cost_basis=None,
                    fees=commission, slippage=slippage_amount,
                ))

            elif order.side == Side.BUY and is_cover:
                # Cover a short position (buy back borrowed shares)
                pos = portfolio.get_position(order.symbol)
                if pos is not None and pos.is_short:
                    avg_cost = pos.avg_entry_price
                    trades = pos.close_lots_fifo(quantity, fill_price, current_date, commission)
                    portfolio.trade_log.extend(trades)
                    portfolio.cash -= fill_price * quantity + commission
                    portfolio.activity_log.append(TradeLogEntry(
                        date=current_date, symbol=order.symbol, action=Side.BUY,
                        quantity=quantity, price=fill_price,
                        value=quantity * fill_price, avg_cost_basis=avg_cost,
                        fees=commission, slippage=slippage_amount,
                    ))
                    if pos.total_quantity == 0:
                        portfolio.close_position(order.symbol)

            elif order.side == Side.SELL and is_short_entry:
                # Short entry: sell shares we don't own, creating negative position
                pos = portfolio.open_position(order.symbol)
                pos.add_lot(-quantity, fill_price, current_date, commission)
                portfolio.cash += fill_price * quantity - commission
                portfolio.activity_log.append(TradeLogEntry(
                    date=current_date, symbol=order.symbol, action=Side.SELL,
                    quantity=quantity, price=fill_price,
                    value=quantity * fill_price, avg_cost_basis=None,
                    fees=commission, slippage=slippage_amount,
                ))

            elif order.side == Side.SELL:
                # Normal long sell
                pos = portfolio.get_position(order.symbol)
                if pos is not None:
                    avg_cost = pos.avg_entry_price
                    trades = pos.sell_lots_fifo(quantity, fill_price, current_date, commission)
                    portfolio.trade_log.extend(trades)
                    portfolio.cash += fill_price * quantity - commission
                    portfolio.activity_log.append(TradeLogEntry(
                        date=current_date, symbol=order.symbol, action=Side.SELL,
                        quantity=quantity, price=fill_price,
                        value=quantity * fill_price, avg_cost_basis=avg_cost,
                        fees=commission, slippage=slippage_amount,
                    ))
                    if pos.total_quantity == 0:
                        portfolio.close_position(order.symbol)

            order.status = OrderStatus.FILLED
            fills.append(fill)
            logger.debug(f"Filled: {fill.side.name} {fill.quantity} {fill.symbol} @ {fill.price:.2f}")

            # Gap 13: Activate bracket children when entry fills
            if order.id in self._bracket_children:
                children = self._bracket_children.pop(order.id)
                self._active_brackets[order.id] = children
                for child in children:
                    remaining_orders.append(child)

            # Gap 13: OCO cancellation — when a bracketed order fills, cancel siblings
            if order.parent_id and order.parent_id in self._active_brackets:
                siblings = self._active_brackets.pop(order.parent_id)
                for sibling in siblings:
                    if sibling.id != order.id:
                        sibling.status = OrderStatus.CANCELLED
                        # Remove from remaining if present
                        if sibling in remaining_orders:
                            remaining_orders.remove(sibling)

        self._pending_orders = remaining_orders
        return fills

    def _handle_unfilled_order(
        self, order: Order, current_date: date, remaining: list[Order]
    ) -> None:
        """Handle an order that could not fill today (limit not reached).

        DAY orders are cancelled.  GTC orders are kept pending with
        ``days_pending`` incremented, unless they have passed their
        ``expiry_date``.
        """
        if order.time_in_force == "GTC":
            # Check explicit expiry
            if order.expiry_date is not None and current_date >= order.expiry_date:
                order.status = OrderStatus.CANCELLED
                order.reason = order.reason or "expired"
                logger.debug(
                    f"GTC order expired: {order.side.name} {order.quantity} "
                    f"{order.symbol} (expiry {order.expiry_date})"
                )
                return
            order.days_pending += 1
            remaining.append(order)
            logger.debug(
                f"GTC order pending (day {order.days_pending}): "
                f"{order.side.name} {order.quantity} {order.symbol}"
            )
        else:
            # DAY order: cancel if not filled
            order.status = OrderStatus.CANCELLED
            order.reason = order.reason or "day_expired"
            logger.debug(
                f"DAY order cancelled: {order.side.name} {order.quantity} {order.symbol}"
            )
