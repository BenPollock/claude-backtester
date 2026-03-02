"""Stop-loss, take-profit, and trailing-stop management.

Extracted from engine.py to isolate stop logic from the main orchestrator.
The stop-trigger-bypasses-broker invariant is preserved: stop exits execute
same-day without going through the T+1 broker queue.
"""

import logging
from datetime import date

import pandas as pd

from backtester.config import StopConfig
from backtester.execution.fees import FeeModel
from backtester.portfolio.order import Order, TradeLogEntry
from backtester.portfolio.portfolio import Portfolio
from backtester.portfolio.position import Position
from backtester.types import Side, OrderType

logger = logging.getLogger(__name__)


class StopManager:
    """Manages stop-loss, take-profit, and trailing-stop logic for positions.

    Receives references to the portfolio and fee model so that stop exits
    can mutate portfolio state directly (same-day execution, bypassing broker).
    """

    def __init__(self, stop_config: StopConfig | None, fee_model: FeeModel, lot_method: str = "fifo"):
        self._config = stop_config
        self._fees = fee_model
        self._lot_method = lot_method

    def set_stops_for_fills(self, fills, today_data: dict[str, pd.Series],
                            portfolio: Portfolio) -> None:
        """Set stop-loss/take-profit/trailing-stop on positions that just got a BUY fill."""
        sc = self._config
        if sc is None:
            return
        for fill in fills:
            if fill.side != Side.BUY:
                continue
            pos = portfolio.get_position(fill.symbol)
            if pos is None:
                continue
            entry = fill.price
            ss = pos.stop_state

            # Percentage-based stops
            if sc.stop_loss_pct is not None:
                ss.stop_loss = entry * (1.0 - sc.stop_loss_pct)
            if sc.take_profit_pct is not None:
                ss.take_profit = entry * (1.0 + sc.take_profit_pct)

            # ATR-based stops (use ATR column if available)
            row = today_data.get(fill.symbol)
            if row is not None:
                atr_val = row.get("ATR")
                if atr_val is not None and not pd.isna(atr_val):
                    if sc.stop_loss_atr is not None:
                        atr_stop = entry - sc.stop_loss_atr * atr_val
                        # Use the tighter of pct and ATR stops
                        if ss.stop_loss is not None:
                            ss.stop_loss = max(ss.stop_loss, atr_stop)
                        else:
                            ss.stop_loss = atr_stop
                    if sc.take_profit_atr is not None:
                        atr_target = entry + sc.take_profit_atr * atr_val
                        if ss.take_profit is not None:
                            ss.take_profit = min(ss.take_profit, atr_target)
                        else:
                            ss.take_profit = atr_target

            # Trailing stop
            if sc.trailing_stop_pct is not None:
                ss.trailing_stop_pct = sc.trailing_stop_pct
                ss.trailing_high = entry  # initialize to entry price

    def set_stops_for_short_fills(self, fills, today_data: dict[str, pd.Series],
                                  portfolio: Portfolio) -> None:
        """Set stop-loss/take-profit on positions that just got a SHORT fill.

        For short positions the stop directions are inverted:
        - stop_loss triggers when price RISES above the stop level
        - take_profit triggers when price FALLS below the target level
        """
        sc = self._config
        if sc is None:
            return
        for fill in fills:
            if fill.side != Side.SELL:
                continue
            pos = portfolio.get_position(fill.symbol)
            if pos is None or not pos.is_short:
                continue
            entry = fill.price
            ss = pos.stop_state

            # For shorts: stop_loss is ABOVE entry (price rises = loss)
            if sc.stop_loss_pct is not None:
                ss.stop_loss = entry * (1.0 + sc.stop_loss_pct)
            # For shorts: take_profit is BELOW entry (price falls = profit)
            if sc.take_profit_pct is not None:
                ss.take_profit = entry * (1.0 - sc.take_profit_pct)

    def check_stop_triggers(self, day: date, today_data: dict[str, pd.Series],
                            portfolio: Portfolio) -> None:
        """Check intraday H/L for stop-loss, take-profit, trailing-stop triggers.

        For long positions:
        - stop_loss/trailing_stop trigger when Low <= stop level
        - take_profit triggers when High >= target level

        For short positions:
        - stop_loss triggers when High >= stop level (price rises against us)
        - take_profit triggers when Low <= target level (price falls in our favor)
        """
        symbols_to_close: list[tuple[str, float, str]] = []
        for symbol, pos in list(portfolio.positions.items()):
            if pos.total_quantity == 0:
                continue
            row = today_data.get(symbol)
            if row is None:
                continue

            low = row.get("Low")
            high = row.get("High")
            if low is None or high is None or pd.isna(low) or pd.isna(high):
                continue

            ss = pos.stop_state
            trigger_price = None
            reason = ""

            if pos.is_short:
                # Short position: stop_loss triggers on HIGH (price rises)
                if ss.stop_loss is not None and high >= ss.stop_loss:
                    trigger_price = ss.stop_loss
                    reason = "stop_loss"

                # Short position: take_profit triggers on LOW (price falls)
                if trigger_price is None:
                    if ss.take_profit is not None and low <= ss.take_profit:
                        trigger_price = ss.take_profit
                        reason = "take_profit"
            else:
                # Long position: original logic
                # Check stop-loss (low touches or breaches stop level)
                if ss.stop_loss is not None and low <= ss.stop_loss:
                    trigger_price = ss.stop_loss
                    reason = "stop_loss"

                # Check trailing stop
                if trigger_price is None:
                    tsp = ss.trailing_stop_price
                    if tsp is not None and low <= tsp:
                        trigger_price = tsp
                        reason = "trailing_stop"

                # Check take-profit (high touches or breaches target)
                if trigger_price is None:
                    if ss.take_profit is not None and high >= ss.take_profit:
                        trigger_price = ss.take_profit
                        reason = "take_profit"

            if trigger_price is not None:
                symbols_to_close.append((symbol, trigger_price, reason))

        # Execute stop exits immediately (same-day, no T+1 delay for stops)
        for symbol, price, reason in symbols_to_close:
            pos = portfolio.get_position(symbol)
            if pos is None or pos.total_quantity == 0:
                continue

            if pos.is_short:
                # Cover the short position
                qty = abs(pos.total_quantity)
                commission = self._fees.compute(
                    Order(symbol=symbol, side=Side.BUY, quantity=qty,
                          order_type=OrderType.STOP, signal_date=day),
                    price, qty
                )
                avg_cost = pos.avg_entry_price
                trades = pos.close_lots_fifo(qty, price, day, commission)
                portfolio.trade_log.extend(trades)
                portfolio.cash -= price * qty + commission  # pay to buy back
                portfolio.activity_log.append(TradeLogEntry(
                    date=day, symbol=symbol, action=Side.BUY,
                    quantity=qty, price=price,
                    value=qty * price, avg_cost_basis=avg_cost,
                    fees=commission, slippage=0.0,
                ))
                portfolio.close_position(symbol)
                logger.debug(f"Short stop triggered ({reason}): {symbol} @ {price:.2f}")
            else:
                # Close the long position using configured lot method
                qty = pos.total_quantity
                commission = self._fees.compute(
                    Order(symbol=symbol, side=Side.SELL, quantity=qty,
                          order_type=OrderType.STOP, signal_date=day),
                    price, qty
                )
                avg_cost = pos.avg_entry_price
                if self._lot_method == "lifo":
                    trades = pos.sell_lots_lifo(qty, price, day, commission)
                elif self._lot_method == "highest_cost":
                    trades = pos.sell_lots_by_cost(qty, price, day, commission, highest_first=True)
                elif self._lot_method == "lowest_cost":
                    trades = pos.sell_lots_by_cost(qty, price, day, commission, highest_first=False)
                else:
                    trades = pos.sell_lots_fifo(qty, price, day, commission)
                portfolio.trade_log.extend(trades)
                portfolio.cash += price * qty - commission
                portfolio.activity_log.append(TradeLogEntry(
                    date=day, symbol=symbol, action=Side.SELL,
                    quantity=qty, price=price,
                    value=qty * price, avg_cost_basis=avg_cost,
                    fees=commission, slippage=0.0,
                ))
                portfolio.close_position(symbol)
                logger.debug(f"Stop triggered ({reason}): {symbol} @ {price:.2f}")

    def update_trailing_highs(self, portfolio: Portfolio,
                              today_data: dict[str, pd.Series]) -> None:
        """Update trailing stop high-water marks using today's High price.

        Called after position market prices are updated to Close.
        """
        for symbol, pos in portfolio.positions.items():
            if pos.stop_state.trailing_stop_pct is None:
                continue
            row = today_data.get(symbol)
            if row is None:
                continue
            high = row.get("High", row.get("Close"))
            if high is not None and not pd.isna(high):
                pos.stop_state.trailing_high = max(
                    pos.stop_state.trailing_high, high
                )
