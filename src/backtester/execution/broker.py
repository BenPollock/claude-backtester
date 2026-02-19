"""Simulated broker: manages pending orders and fills them at market open."""

import logging
from datetime import date

import pandas as pd

from backtester.portfolio.order import Order, Fill, TradeLogEntry
from backtester.portfolio.portfolio import Portfolio
from backtester.execution.slippage import SlippageModel, FixedSlippage
from backtester.execution.fees import FeeModel, PerTradeFee
from backtester.types import Side, OrderStatus

logger = logging.getLogger(__name__)


class SimulatedBroker:
    """Accepts orders and fills them at the next day's open price + slippage + fees."""

    def __init__(self, slippage: SlippageModel | None = None, fees: FeeModel | None = None):
        self._slippage = slippage or FixedSlippage()
        self._fees = fees or PerTradeFee()
        self._pending_orders: list[Order] = []

    def submit_order(self, order: Order) -> None:
        """Queue an order for fill on the next trading day."""
        self._pending_orders.append(order)
        logger.debug(f"Order submitted: {order.side.name} {order.quantity} {order.symbol}")

    @property
    def pending_orders(self) -> list[Order]:
        return list(self._pending_orders)

    def process_fills(
        self,
        current_date: date,
        market_data: dict[str, pd.Series],
        portfolio: Portfolio,
    ) -> list[Fill]:
        """Fill pending orders at today's OPEN price + slippage.

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
                # No market data — keep order pending
                remaining_orders.append(order)
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
                quantity = pos.total_quantity

            # Validate BUY: enough cash?
            fill_price = self._slippage.compute(order, open_price, volume)
            commission = self._fees.compute(order, fill_price, quantity)

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

            # Execute fill
            slippage_amount = abs(fill_price - open_price)
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

            # Update portfolio
            if order.side == Side.BUY:
                pos = portfolio.open_position(order.symbol)
                pos.add_lot(quantity, fill_price, current_date, commission)
                portfolio.cash -= fill_price * quantity + commission
                portfolio.activity_log.append(TradeLogEntry(
                    date=current_date, symbol=order.symbol, action=Side.BUY,
                    quantity=quantity, price=fill_price,
                    value=quantity * fill_price, avg_cost_basis=None,
                    fees=commission, slippage=slippage_amount,
                ))

            elif order.side == Side.SELL:
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

        self._pending_orders = remaining_orders
        return fills
