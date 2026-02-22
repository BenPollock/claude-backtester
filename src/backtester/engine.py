"""Backtest engine: orchestrates data loading, strategy execution, and analytics."""

import logging
from datetime import date

import pandas as pd

from backtester.config import BacktestConfig
from backtester.data.manager import DataManager
from backtester.data.calendar import TradingCalendar
from backtester.strategies.base import Strategy
from backtester.strategies.registry import get_strategy
from backtester.strategies.indicators import sma
from backtester.execution.broker import SimulatedBroker
from backtester.execution.slippage import FixedSlippage, VolumeSlippage
from backtester.execution.fees import PerTradeFee
from backtester.execution.position_sizing import (
    PositionSizer, FixedFractional, ATRSizer, VolatilityParity,
)
from backtester.portfolio.portfolio import Portfolio
from backtester.portfolio.order import Order
from backtester.portfolio.position import StopState
from backtester.types import Side, OrderType, SignalAction

logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtest outputs."""

    def __init__(self, config: BacktestConfig, portfolio: Portfolio,
                 benchmark_equity: list[tuple[date, float]] | None = None):
        self.config = config
        self.portfolio = portfolio
        self.benchmark_equity = benchmark_equity

    @property
    def equity_series(self) -> pd.Series:
        dates, values = zip(*self.portfolio.equity_history)
        return pd.Series(values, index=pd.DatetimeIndex(dates, name="Date"), name="Equity")

    @property
    def benchmark_series(self) -> pd.Series | None:
        if not self.benchmark_equity:
            return None
        dates, values = zip(*self.benchmark_equity)
        return pd.Series(values, index=pd.DatetimeIndex(dates, name="Date"), name="Benchmark")

    @property
    def trades(self):
        return self.portfolio.trade_log

    @property
    def activity_log(self):
        return self.portfolio.activity_log


class BacktestEngine:
    """Main backtest orchestrator.

    Flow:
    1. Load universe data (cache-first)
    2. Compute indicators per ticker (vectorized)
    3. Load benchmark + compute benchmark indicators
    4. Iterate trading days: fill orders, update prices, generate signals
    5. Force-close all positions on last day
    6. Return BacktestResult for analytics
    """

    def __init__(self, config: BacktestConfig, data_manager: DataManager | None = None):
        self.config = config
        self._data = data_manager or DataManager(cache_dir=config.data_cache_dir)
        self._calendar = TradingCalendar()

        # Set up execution models
        if config.slippage_model == "volume":
            slippage = VolumeSlippage()
        else:
            slippage = FixedSlippage(bps=config.slippage_bps)

        fees = PerTradeFee(fee=config.fee_per_trade)
        self._broker = SimulatedBroker(slippage=slippage, fees=fees)

        # Set up position sizer
        if config.position_sizing == "atr":
            self._sizer: PositionSizer = ATRSizer(
                risk_pct=config.sizing_risk_pct,
                atr_multiple=config.sizing_atr_multiple,
            )
        elif config.position_sizing == "vol_parity":
            self._sizer = VolatilityParity()
        else:
            self._sizer = FixedFractional()

    def run(self) -> BacktestResult:
        """Execute the full backtest and return results."""
        config = self.config

        # 1. Load strategy
        strategy = get_strategy(config.strategy_name)
        strategy.configure(config.strategy_params)

        # 2. Load universe data
        logger.info(f"Loading data for {len(config.tickers)} tickers...")
        universe_data = self._data.load_many(config.tickers, config.start_date, config.end_date)

        if not universe_data:
            raise RuntimeError("No data loaded for any ticker")

        # 3. Compute indicators per ticker
        logger.info("Computing indicators...")
        for symbol in list(universe_data.keys()):
            universe_data[symbol] = strategy.compute_indicators(universe_data[symbol])

        # 4. Load benchmark
        benchmark_data = None
        if config.benchmark:
            benchmark_data = self._data.load(config.benchmark, config.start_date, config.end_date)
            # Compute benchmark indicators for regime filter
            if config.regime_filter:
                benchmark_data = self._compute_regime_indicators(benchmark_data)
            benchmark_data = strategy.compute_benchmark_indicators(benchmark_data)

        # 5. Get trading days
        trading_days = self._calendar.trading_days(config.start_date, config.end_date)

        # 6. Initialize portfolio
        portfolio = Portfolio(cash=config.starting_cash)

        # 7. Track benchmark equity for comparison
        benchmark_equity: list[tuple[date, float]] = []
        benchmark_shares = 0

        # 8. Main loop
        logger.info(f"Running backtest: {len(trading_days)} trading days, {len(universe_data)} tickers")

        for i, ts in enumerate(trading_days):
            day = ts.date() if hasattr(ts, 'date') else ts
            today_data: dict[str, pd.Series] = {}

            # Collect today's rows for all symbols
            for symbol, df in universe_data.items():
                if ts in df.index:
                    today_data[symbol] = df.loc[ts]

            # a. Process fills from yesterday's orders at today's open
            fills = self._broker.process_fills(day, today_data, portfolio)

            # a2. Set stop levels on newly opened positions
            if config.stop_config:
                self._set_stops_for_fills(fills, today_data, portfolio)

            # a3. Check stop-loss / take-profit / trailing stop triggers using intraday H/L
            if config.stop_config:
                self._check_stop_triggers(day, today_data, portfolio)

            # b. Update position market prices to today's close
            for symbol, pos in portfolio.positions.items():
                row = today_data.get(symbol)
                if row is not None and not pd.isna(row.get("Close")):
                    pos.update_market_price(row["Close"])
                    # Update trailing stop high-water mark
                    if pos.stop_state.trailing_stop_pct is not None:
                        high = row.get("High", row["Close"])
                        if not pd.isna(high):
                            pos.stop_state.trailing_high = max(
                                pos.stop_state.trailing_high, high
                            )

            # c. Record equity
            portfolio.record_equity(day)

            # d. Track benchmark equity
            if benchmark_data is not None and ts in benchmark_data.index:
                bm_row = benchmark_data.loc[ts]
                bm_close = bm_row.get("Close")
                if not pd.isna(bm_close):
                    if benchmark_shares == 0 and bm_close > 0:
                        benchmark_shares = config.starting_cash / bm_close
                    benchmark_equity.append((day, benchmark_shares * bm_close))

            # e. Check regime filter
            regime_on = True
            if config.regime_filter and benchmark_data is not None and ts in benchmark_data.index:
                regime_on = self._check_regime(benchmark_data.loc[ts])

            # f. Generate signals (skip last day — we'll force-close)
            if i < len(trading_days) - 1:
                benchmark_row = None
                if benchmark_data is not None and ts in benchmark_data.index:
                    benchmark_row = benchmark_data.loc[ts]

                portfolio_state = portfolio.snapshot()

                for symbol, row in today_data.items():
                    if pd.isna(row.get("Close")):
                        continue

                    position = portfolio.get_position(symbol)
                    signal = strategy.generate_signals(
                        symbol, row, position, portfolio_state, benchmark_row
                    )

                    # Regime filter: suppress BUY signals when regime is off
                    if signal == SignalAction.BUY and not regime_on:
                        signal = SignalAction.HOLD

                    if signal == SignalAction.HOLD:
                        continue

                    # Check position limits for BUY
                    if signal == SignalAction.BUY:
                        if portfolio_state.num_positions >= config.max_positions:
                            continue
                        if symbol in portfolio_state.position_symbols:
                            continue  # already have a position

                    # Size the order
                    if signal == SignalAction.BUY:
                        qty = self._sizer.compute(
                            symbol, row["Close"], row,
                            portfolio_state.total_equity,
                            portfolio_state.cash,
                            config.max_alloc_pct,
                        )
                    else:
                        qty = strategy.size_order(
                            symbol, signal, row, portfolio_state, config.max_alloc_pct
                        )

                    if qty == 0:
                        continue

                    side = Side.BUY if signal == SignalAction.BUY else Side.SELL
                    order = Order(
                        symbol=symbol,
                        side=side,
                        quantity=qty,
                        order_type=OrderType.MARKET,
                        signal_date=day,
                    )
                    self._broker.submit_order(order)

        # 9. Force-close all positions on last day
        last_day = trading_days[-1]
        last_date = last_day.date() if hasattr(last_day, 'date') else last_day
        self._force_close_all(portfolio, last_date)

        logger.info(f"Backtest complete. {len(portfolio.trade_log)} trades executed.")

        return BacktestResult(
            config=config,
            portfolio=portfolio,
            benchmark_equity=benchmark_equity if benchmark_equity else None,
        )

    def _compute_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime filter indicator columns to benchmark data."""
        rf = self.config.regime_filter
        if rf is None:
            return df
        df = df.copy()
        if rf.indicator == "sma":
            df["regime_fast"] = sma(df["Close"], rf.fast_period)
            df["regime_slow"] = sma(df["Close"], rf.slow_period)
        return df

    def _check_regime(self, benchmark_row: pd.Series) -> bool:
        """Return True if regime filter allows BUY signals."""
        rf = self.config.regime_filter
        if rf is None:
            return True

        fast = benchmark_row.get("regime_fast")
        slow = benchmark_row.get("regime_slow")

        if pd.isna(fast) or pd.isna(slow):
            return True  # allow trading during warmup

        if rf.condition == "fast_above_slow":
            return fast > slow

        return True  # unknown condition — default to allowing trades

    def _set_stops_for_fills(self, fills, today_data, portfolio):
        """Set stop-loss/take-profit/trailing-stop on positions that just got a BUY fill."""
        sc = self.config.stop_config
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

    def _check_stop_triggers(self, day, today_data, portfolio):
        """Check intraday H/L for stop-loss, take-profit, trailing-stop triggers.

        Uses intraday Low for stop-loss/trailing-stop and High for take-profit
        to determine if the stop would have been hit during the day.
        Stop fills use the stop price (not open), which is more realistic.
        """
        symbols_to_close = []
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
        from backtester.portfolio.order import TradeLogEntry
        for symbol, price, reason in symbols_to_close:
            pos = portfolio.get_position(symbol)
            if pos is None or pos.total_quantity == 0:
                continue
            qty = pos.total_quantity
            commission = self._broker._fees.compute(
                Order(symbol=symbol, side=Side.SELL, quantity=qty,
                      order_type=OrderType.STOP, signal_date=day),
                price, qty
            )
            avg_cost = pos.avg_entry_price
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

    def _force_close_all(self, portfolio: Portfolio, last_date: date) -> None:
        """Close all remaining positions at their last known market price."""
        for symbol in list(portfolio.positions.keys()):
            pos = portfolio.positions[symbol]
            if pos.total_quantity > 0:
                price = pos._market_price
                if price > 0:
                    trades = pos.sell_lots_fifo(pos.total_quantity, price, last_date, 0.0)
                    portfolio.trade_log.extend(trades)
                    portfolio.cash += price * sum(t.quantity for t in trades)
                    portfolio.close_position(symbol)
                    logger.debug(f"Force-closed {symbol}: {len(trades)} lots")
