"""Backtest engine: orchestrates data loading, strategy execution, and analytics."""

import logging
from datetime import date

import pandas as pd

from backtester.config import BacktestConfig
from backtester.data.manager import DataManager, resample_ohlcv
from backtester.data.calendar import TradingCalendar
from backtester.strategies.base import Signal, Strategy
from backtester.strategies.registry import get_strategy
from backtester.strategies.indicators import sma
from backtester.execution.broker import SimulatedBroker
from backtester.execution.slippage import FixedSlippage, VolumeSlippage
from backtester.execution.fees import PerTradeFee
from backtester.execution.position_sizing import (
    PositionSizer, FixedFractional, ATRSizer, VolatilityParity,
)
from backtester.execution.stops import StopManager
from backtester.portfolio.portfolio import Portfolio
from backtester.portfolio.order import Order
from backtester.result import BacktestResult
from backtester.types import Side, OrderType, SignalAction

# Re-export BacktestResult for backward compatibility
__all__ = ["BacktestEngine", "BacktestResult"]

logger = logging.getLogger(__name__)


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
        self._stop_mgr = StopManager(config.stop_config, fees)

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

        # 3. Resample data for multi-timeframe strategies and compute indicators
        extra_timeframes = [tf for tf in strategy.timeframes if tf != "daily"]

        # Build per-symbol timeframe_data dicts (resampled + forward-filled to daily index)
        symbol_tf_data: dict[str, dict[str, pd.DataFrame]] = {}
        if extra_timeframes:
            logger.info(f"Resampling data for timeframes: {extra_timeframes}")
            for symbol, daily_df in universe_data.items():
                tf_map: dict[str, pd.DataFrame] = {}
                for tf in extra_timeframes:
                    resampled = resample_ohlcv(daily_df, tf)
                    # Forward-fill resampled data onto the daily index so
                    # strategies can look up the current period value for any day.
                    ff = resampled.reindex(daily_df.index).ffill()
                    tf_map[tf] = ff
                symbol_tf_data[symbol] = tf_map

        logger.info("Computing indicators...")
        for symbol in list(universe_data.keys()):
            tf_data = symbol_tf_data.get(symbol) if extra_timeframes else None
            universe_data[symbol] = strategy.compute_indicators(
                universe_data[symbol], timeframe_data=tf_data
            )

            # Merge forward-filled timeframe columns into the daily DataFrame
            # with a prefix (e.g. "weekly_Close") so generate_signals() can
            # access multi-timeframe values directly from the daily row.
            if tf_data:
                for tf_name, tf_df in tf_data.items():
                    for col in tf_df.columns:
                        universe_data[symbol][f"{tf_name}_{col}"] = tf_df[col]

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
                self._stop_mgr.set_stops_for_fills(fills, today_data, portfolio)

            # a3. Check stop-loss / take-profit / trailing stop triggers using intraday H/L
            if config.stop_config:
                self._stop_mgr.check_stop_triggers(day, today_data, portfolio)

            # b. Update position market prices to today's close
            for symbol, pos in portfolio.positions.items():
                row = today_data.get(symbol)
                if row is not None and not pd.isna(row.get("Close")):
                    pos.update_market_price(row["Close"])

            # b1. Accrue short borrow costs (after market price update)
            if config.allow_short and config.short_borrow_rate > 0:
                for symbol, pos in portfolio.positions.items():
                    if pos.is_short:
                        pos.accrue_borrow_cost(config.short_borrow_rate)

            # b2. Update trailing stop high-water marks
            if config.stop_config:
                self._stop_mgr.update_trailing_highs(portfolio, today_data)

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
                    raw_signal = strategy.generate_signals(
                        symbol, row, position, portfolio_state, benchmark_row
                    )

                    # Unwrap Signal object to extract action and limit params
                    if isinstance(raw_signal, Signal):
                        signal = raw_signal.action
                        limit_price = raw_signal.limit_price
                        time_in_force = raw_signal.time_in_force
                        expiry_date = raw_signal.expiry_date
                    else:
                        signal = raw_signal
                        limit_price = None
                        time_in_force = "DAY"
                        expiry_date = None

                    # Regime filter: suppress BUY signals when regime is off
                    if signal == SignalAction.BUY and not regime_on:
                        signal = SignalAction.HOLD

                    # Regime filter: optionally suppress SHORT signals too
                    if signal == SignalAction.SHORT and not regime_on:
                        signal = SignalAction.HOLD

                    # Reject SHORT signals if short selling is disabled
                    if signal == SignalAction.SHORT and not config.allow_short:
                        signal = SignalAction.HOLD

                    # Reject COVER if no short position exists
                    if signal == SignalAction.COVER:
                        if position is None or not position.is_short:
                            signal = SignalAction.HOLD

                    if signal == SignalAction.HOLD:
                        continue

                    # Check position limits for BUY
                    if signal == SignalAction.BUY:
                        if portfolio_state.num_positions >= config.max_positions:
                            continue
                        if symbol in portfolio_state.position_symbols:
                            continue  # already have a position

                    # Check position limits for SHORT
                    if signal == SignalAction.SHORT:
                        if portfolio_state.num_positions >= config.max_positions:
                            continue
                        if symbol in portfolio_state.position_symbols:
                            continue  # already have a position (long or short)

                    # Size the order
                    if signal == SignalAction.BUY:
                        qty = self._sizer.compute(
                            symbol, row["Close"], row,
                            portfolio_state.total_equity,
                            portfolio_state.cash,
                            config.max_alloc_pct,
                        )
                    elif signal == SignalAction.SHORT:
                        # Use strategy's size_order for SHORT (returns negative qty)
                        qty = strategy.size_order(
                            symbol, signal, row, portfolio_state, config.max_alloc_pct
                        )
                    else:
                        # SELL or COVER
                        qty = strategy.size_order(
                            symbol, signal, row, portfolio_state, config.max_alloc_pct
                        )

                    if qty == 0:
                        continue

                    # Map signal to Side and construct order
                    if signal == SignalAction.BUY:
                        side = Side.BUY
                        reason = ""
                    elif signal == SignalAction.SELL:
                        side = Side.SELL
                        reason = ""
                    elif signal == SignalAction.SHORT:
                        side = Side.SELL
                        reason = "short_entry"
                        qty = abs(qty)  # broker expects positive qty for short entry
                    elif signal == SignalAction.COVER:
                        side = Side.BUY
                        reason = "cover"
                        # qty is -1 sentinel; broker will resolve
                    else:
                        continue

                    # Determine order type based on limit_price
                    order_type = OrderType.LIMIT if limit_price is not None else OrderType.MARKET

                    order = Order(
                        symbol=symbol,
                        side=side,
                        quantity=qty,
                        order_type=order_type,
                        signal_date=day,
                        limit_price=limit_price,
                        time_in_force=time_in_force,
                        expiry_date=expiry_date,
                        reason=reason,
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

    def _force_close_all(self, portfolio: Portfolio, last_date: date) -> None:
        """Close all remaining positions at their last known market price."""
        for symbol in list(portfolio.positions.keys()):
            pos = portfolio.positions[symbol]
            if pos.total_quantity > 0:
                # Long position: sell at market price
                price = pos._market_price
                if price > 0:
                    trades = pos.sell_lots_fifo(pos.total_quantity, price, last_date, 0.0)
                    portfolio.trade_log.extend(trades)
                    portfolio.cash += price * sum(t.quantity for t in trades)
                    portfolio.close_position(symbol)
                    logger.debug(f"Force-closed long {symbol}: {len(trades)} lots")
            elif pos.total_quantity < 0:
                # Short position: cover at market price
                price = pos._market_price
                if price > 0:
                    cover_qty = abs(pos.total_quantity)
                    trades = pos.close_lots_fifo(cover_qty, price, last_date, 0.0)
                    portfolio.trade_log.extend(trades)
                    portfolio.cash -= price * cover_qty  # pay to buy back
                    portfolio.close_position(symbol)
                    logger.debug(f"Force-closed short {symbol}: {len(trades)} lots")
