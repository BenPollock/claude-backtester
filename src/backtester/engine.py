"""Backtest engine: orchestrates data loading, strategy execution, and analytics."""

import logging
import sys
from datetime import date

import pandas as pd

import math
from tqdm import tqdm

from backtester.config import BacktestConfig
from backtester.data.manager import DataManager, resample_ohlcv
from backtester.data.calendar import TradingCalendar
from backtester.data.universe import HistoricalUniverse
from backtester.strategies.base import Signal, Strategy
from backtester.strategies.registry import get_strategy
from backtester.strategies.indicators import sma
from backtester.execution.broker import SimulatedBroker
from backtester.execution.slippage import FixedSlippage, VolumeSlippage, SqrtImpactSlippage
from backtester.execution.fees import PerTradeFee, PercentageFee, CompositeFee, SECFee, TAFFee
from backtester.execution.position_sizing import (
    PositionSizer, FixedFractional, ATRSizer, VolatilityParity,
    KellyCriterionSizer, RiskParitySizer,
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
        self._data = data_manager or DataManager(
            cache_dir=config.data_cache_dir,
            source=self._create_data_source(config),
        )
        self._calendar = TradingCalendar()

        # Set up execution models
        if config.slippage_model == "volume":
            slippage = VolumeSlippage(impact_factor=config.slippage_impact_factor)
        elif config.slippage_model == "sqrt":
            slippage = SqrtImpactSlippage(impact_factor=config.slippage_impact_factor)
        else:
            slippage = FixedSlippage(bps=config.slippage_bps)

        # Fee model factory
        if config.fee_model == "percentage":
            fees = PercentageFee(bps=config.fee_per_trade)
        elif config.fee_model == "composite_us":
            fees = CompositeFee([
                PercentageFee(bps=config.fee_per_trade),
                SECFee(),
                TAFFee(),
            ])
        else:
            fees = PerTradeFee(fee=config.fee_per_trade)
        self._broker = SimulatedBroker(
            slippage=slippage, fees=fees,
            max_volume_pct=config.max_volume_pct,
            partial_fill_policy=config.partial_fill_policy,
            fill_price_model=config.fill_price_model,
        )
        self._stop_mgr = StopManager(config.stop_config, fees, lot_method=config.lot_method)

        # Set up position sizer
        if config.position_sizing == "atr":
            self._sizer: PositionSizer = ATRSizer(
                risk_pct=config.sizing_risk_pct,
                atr_multiple=config.sizing_atr_multiple,
            )
        elif config.position_sizing == "vol_parity":
            self._sizer = VolatilityParity(
                target_vol=config.sizing_target_vol,
                lookback=config.sizing_vol_lookback,
            )
        elif config.position_sizing == "kelly":
            self._sizer = KellyCriterionSizer(fraction=config.kelly_fraction)
        elif config.position_sizing == "risk_parity":
            self._sizer = RiskParitySizer(
                target_vol=config.sizing_target_vol,
            )
        else:
            self._sizer = FixedFractional()

        # Gap 1: Historical universe
        self._historical_universe: HistoricalUniverse | None = None
        if config.universe_file:
            self._historical_universe = HistoricalUniverse(config.universe_file)

        # Gap 5: Drawdown kill switch
        self._halted = False

        # Gap 2: Delisting detection — track consecutive missing days per held symbol
        self._missing_days: dict[str, int] = {}

        # Gap 18: Sector map
        self._sector_map: dict[str, str] = {}
        if config.sector_map_path:
            self._load_sector_map(config.sector_map_path)

    @staticmethod
    def _create_data_source(config: BacktestConfig):
        """Create appropriate data source based on config."""
        if config.data_source == "csv":
            from backtester.data.csv_source import CSVDataSource
            return CSVDataSource(config.data_path or ".")
        elif config.data_source == "parquet":
            from backtester.data.parquet_source import ParquetDataSource
            return ParquetDataSource(config.data_path or ".")
        # Default: yahoo (handled by DataManager default)
        return None

    def _load_sector_map(self, path: str) -> None:
        """Load sector map from CSV (symbol,sector)."""
        import csv
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = row["symbol"].strip().upper()
                sector = row["sector"].strip()
                self._sector_map[sym] = sector

    def run(self) -> BacktestResult:
        """Execute the full backtest and return results."""
        config = self.config

        # 1. Load strategy
        strategy = get_strategy(config.strategy_name)
        strategy.configure(config.strategy_params)

        # Gap 9: Inject fundamental data manager if configured
        if config.fundamental_data_path:
            from backtester.data.fundamental import FundamentalDataManager
            fm = FundamentalDataManager(config.fundamental_data_path)
            strategy.set_fundamental_data(fm)

        # 2. Load universe data — include historical universe symbols
        tickers = list(config.tickers)
        if self._historical_universe:
            extra = self._historical_universe.all_symbols - set(tickers)
            tickers.extend(sorted(extra))

        logger.info(f"Loading data for {len(tickers)} tickers...")
        universe_data = self._data.load_many(tickers, config.start_date, config.end_date)

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
                    ff = resampled.reindex(daily_df.index).ffill()
                    tf_map[tf] = ff
                symbol_tf_data[symbol] = tf_map

        logger.info("Computing indicators...")
        for symbol in list(universe_data.keys()):
            tf_data = symbol_tf_data.get(symbol) if extra_timeframes else None
            universe_data[symbol] = strategy.compute_indicators(
                universe_data[symbol], timeframe_data=tf_data
            )

            if tf_data:
                for tf_name, tf_df in tf_data.items():
                    for col in tf_df.columns:
                        universe_data[symbol][f"{tf_name}_{col}"] = tf_df[col]

        # 4. Load benchmark
        benchmark_data = None
        if config.benchmark:
            benchmark_data = self._data.load(config.benchmark, config.start_date, config.end_date)
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

        # Gap 20: vol targeting state
        prev_day_date = None

        # 8. Main loop
        logger.info(f"Running backtest: {len(trading_days)} trading days, {len(universe_data)} tickers")

        day_iter = tqdm(
            enumerate(trading_days),
            total=len(trading_days),
            desc="Backtesting",
            disable=not sys.stderr.isatty(),
        )
        for i, ts in day_iter:
            day = ts.date() if hasattr(ts, 'date') else ts
            today_data: dict[str, pd.Series] = {}

            for symbol, df in universe_data.items():
                if ts in df.index:
                    today_data[symbol] = df.loc[ts]

            # Gap 1: Filter by historical universe membership
            if self._historical_universe:
                members = self._historical_universe.members_on(day)
                if members:
                    today_data = {s: r for s, r in today_data.items() if s in members}

            # a. Process fills from yesterday's orders at today's open
            fills = self._broker.process_fills(day, today_data, portfolio)

            # a2. Set stop levels on newly opened positions
            if config.stop_config:
                self._stop_mgr.set_stops_for_fills(fills, today_data, portfolio)

            # a3. Check stop triggers using intraday H/L
            if config.stop_config:
                self._stop_mgr.check_stop_triggers(day, today_data, portfolio)

            # Gap 2: Delisting detection — force-close held symbols missing >5 days
            if portfolio.positions:
                self._check_delistings(day, today_data, portfolio)

            # b. Update position market prices to today's close
            for symbol, pos in portfolio.positions.items():
                row = today_data.get(symbol)
                if row is not None and not pd.isna(row.get("Close")):
                    pos.update_market_price(row["Close"])

            # b1. Accrue short borrow costs and deduct from cash (Gap 35)
            if config.allow_short and config.short_borrow_rate > 0:
                for symbol, pos in portfolio.positions.items():
                    if pos.is_short:
                        cost = pos.accrue_borrow_cost(config.short_borrow_rate)
                        portfolio.cash -= cost

            # b2. Update trailing stop high-water marks
            if config.stop_config:
                self._stop_mgr.update_trailing_highs(portfolio, today_data)

            # Gap 16: DRIP — reinvest dividends
            if config.drip:
                self._process_drip(day, today_data, portfolio)

            # c. Record equity
            portfolio.record_equity(day)

            # Gap 5: Drawdown kill switch
            if config.max_drawdown_pct is not None and not self._halted:
                equity_vals = [v for _, v in portfolio.equity_history]
                if len(equity_vals) >= 2:
                    peak = max(equity_vals)
                    current = equity_vals[-1]
                    dd = (current - peak) / peak
                    if dd <= -config.max_drawdown_pct:
                        self._force_close_all(portfolio, day)
                        self._halted = True
                        logger.info(f"Drawdown kill switch triggered at {dd:.2%} on {day}")

            # d. Track benchmark equity
            if benchmark_data is not None and ts in benchmark_data.index:
                bm_row = benchmark_data.loc[ts]
                bm_close = bm_row.get("Close")
                if not pd.isna(bm_close):
                    if benchmark_shares == 0 and bm_close > 0:
                        benchmark_shares = config.starting_cash / bm_close
                    benchmark_equity.append((day, benchmark_shares * bm_close))

            # Skip signal generation if halted
            if self._halted:
                prev_day_date = day
                continue

            # e. Check regime filter
            regime_on = True
            if config.regime_filter and benchmark_data is not None and ts in benchmark_data.index:
                regime_on = self._check_regime(benchmark_data.loc[ts])

            # Gap 45: Rebalance schedule — skip signal gen on non-rebalance days
            if config.rebalance_schedule != "daily" and prev_day_date is not None:
                if not self._is_rebalance_day_for_schedule(day, prev_day_date):
                    prev_day_date = day
                    continue

            # Gap 20: Portfolio-level volatility targeting
            vol_scale = 1.0
            if config.target_portfolio_vol is not None:
                vol_scale = self._compute_vol_scale(portfolio, config)

            # f. Generate signals (skip last day — we'll force-close)
            if i < len(trading_days) - 1:
                benchmark_row = None
                if benchmark_data is not None and ts in benchmark_data.index:
                    benchmark_row = benchmark_data.loc[ts]

                portfolio_state = portfolio.snapshot()

                # Gap 4: Cross-sectional strategy dispatch
                from backtester.strategies.base import CrossSectionalStrategy
                if isinstance(strategy, CrossSectionalStrategy):
                    signals_list = strategy.rank_universe(
                        today_data, portfolio.positions, portfolio_state, benchmark_row
                    )
                    # Gap 15: Target-weight rebalancing
                    target_weights = strategy.target_weights(
                        today_data, portfolio_state, benchmark_row
                    )
                    if target_weights is not None:
                        self._process_rebalance(target_weights, today_data, portfolio, day, vol_scale)
                    else:
                        for symbol, signal_action in signals_list:
                            row = today_data.get(symbol)
                            if row is None:
                                continue
                            self._process_signal(
                                symbol, signal_action, None, None, None,
                                row, portfolio, portfolio_state, strategy,
                                config, regime_on, day, vol_scale,
                            )
                else:
                    # Gap 15: Check target_weights for non-cross-sectional too
                    target_weights = strategy.target_weights(
                        today_data, portfolio_state, benchmark_row
                    )
                    if target_weights is not None:
                        self._process_rebalance(target_weights, today_data, portfolio, day, vol_scale)
                    else:
                        for symbol, row in today_data.items():
                            if pd.isna(row.get("Close")):
                                continue

                            position = portfolio.get_position(symbol)
                            raw_signal = strategy.generate_signals(
                                symbol, row, position, portfolio_state, benchmark_row
                            )

                            # Unwrap Signal object
                            if isinstance(raw_signal, Signal):
                                signal = raw_signal.action
                                limit_price = raw_signal.limit_price
                                time_in_force = raw_signal.time_in_force
                                expiry_date = raw_signal.expiry_date
                                stop_price = raw_signal.stop_price
                                order_type = raw_signal.order_type
                            else:
                                signal = raw_signal
                                limit_price = None
                                time_in_force = "DAY"
                                expiry_date = None
                                stop_price = None
                                order_type = OrderType.MARKET

                            self._process_signal(
                                symbol, signal, limit_price, time_in_force,
                                expiry_date, row, portfolio, portfolio_state,
                                strategy, config, regime_on, day, vol_scale,
                                stop_price=stop_price, order_type=order_type,
                            )

            prev_day_date = day

        # 9. Force-close all positions on last day
        if not self._halted:
            last_day = trading_days[-1]
            last_date = last_day.date() if hasattr(last_day, 'date') else last_day
            self._force_close_all(portfolio, last_date)

        logger.info(f"Backtest complete. {len(portfolio.trade_log)} trades executed.")

        # Extract benchmark close prices for analytics
        benchmark_prices = None
        if benchmark_data is not None and "Close" in benchmark_data.columns:
            benchmark_prices = benchmark_data["Close"]

        result = BacktestResult(
            config=config,
            portfolio=portfolio,
            benchmark_equity=benchmark_equity if benchmark_equity else None,
            benchmark_prices=benchmark_prices,
            universe_data=universe_data,
        )

        # Gap 30: Save results if configured
        if config.save_results_path:
            result.save(config.save_results_path)
            logger.info(f"Results saved to {config.save_results_path}")

        return result

    def _process_signal(
        self, symbol, signal, limit_price, time_in_force, expiry_date,
        row, portfolio, portfolio_state, strategy, config, regime_on, day,
        vol_scale=1.0, stop_price=None, order_type=None,
    ) -> None:
        """Process a single signal through filtering, sizing, and order submission."""
        # Regime filter: suppress BUY signals when regime is off
        if signal == SignalAction.BUY and not regime_on:
            signal = SignalAction.HOLD
        if signal == SignalAction.SHORT and not regime_on:
            signal = SignalAction.HOLD

        # Reject SHORT if disabled
        if signal == SignalAction.SHORT and not config.allow_short:
            signal = SignalAction.HOLD

        # Reject COVER if no short position
        if signal == SignalAction.COVER:
            position = portfolio.get_position(symbol)
            if position is None or not position.is_short:
                signal = SignalAction.HOLD

        if signal == SignalAction.HOLD:
            return

        # Check position limits for BUY
        if signal == SignalAction.BUY:
            if portfolio_state.num_positions >= config.max_positions:
                return
            if symbol in portfolio_state.position_symbols:
                return

        # Check position limits for SHORT
        if signal == SignalAction.SHORT:
            if portfolio_state.num_positions >= config.max_positions:
                return
            if symbol in portfolio_state.position_symbols:
                return

        # Gap 18: Sector exposure limit check for BUY/SHORT
        if signal in (SignalAction.BUY, SignalAction.SHORT):
            if config.max_sector_exposure is not None and self._sector_map:
                target_sector = self._sector_map.get(symbol, "Unknown")
                equity = portfolio_state.total_equity
                if equity > 0:
                    sector_value = sum(
                        abs(pos.market_value) for sym, pos in portfolio.positions.items()
                        if self._sector_map.get(sym, "Unknown") == target_sector
                    )
                    close_price = row.get("Close", 0)
                    # Estimate new position value
                    est_new = close_price * int(equity * config.max_alloc_pct / close_price) if close_price > 0 else 0
                    if (sector_value + est_new) / equity >= config.max_sector_exposure:
                        return

        # Gap 19: Gross/net exposure limit check
        if signal in (SignalAction.BUY, SignalAction.SHORT):
            equity = portfolio_state.total_equity
            if equity > 0:
                long_val = sum(pos.market_value for pos in portfolio.positions.values() if not pos.is_short)
                short_val = sum(abs(pos.market_value) for pos in portfolio.positions.values() if pos.is_short)
                gross = (long_val + short_val) / equity
                net = (long_val - short_val) / equity

                if config.max_gross_exposure is not None and gross >= config.max_gross_exposure:
                    return
                if config.max_net_exposure is not None:
                    if signal == SignalAction.BUY and net >= config.max_net_exposure:
                        return

        # Size the order
        if signal == SignalAction.BUY:
            qty = self._sizer.compute(
                symbol, row["Close"], row,
                portfolio_state.total_equity,
                portfolio_state.cash,
                config.max_alloc_pct,
            )
            # Gap 20: Apply vol scale
            if vol_scale != 1.0:
                qty = int(qty * vol_scale)
        elif signal == SignalAction.SHORT:
            qty = strategy.size_order(
                symbol, signal, row, portfolio_state, config.max_alloc_pct
            )
        else:
            qty = strategy.size_order(
                symbol, signal, row, portfolio_state, config.max_alloc_pct
            )

        if qty == 0:
            return

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
            qty = abs(qty)
        elif signal == SignalAction.COVER:
            side = Side.BUY
            reason = "cover"
        else:
            return

        # Determine order type
        if order_type is not None and order_type != OrderType.MARKET:
            final_order_type = order_type
        elif limit_price is not None:
            final_order_type = OrderType.LIMIT
        else:
            final_order_type = OrderType.MARKET

        order = Order(
            symbol=symbol,
            side=side,
            quantity=qty,
            order_type=final_order_type,
            signal_date=day,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force or "DAY",
            expiry_date=expiry_date,
            reason=reason,
        )
        self._broker.submit_order(order)

    def _process_rebalance(
        self, target_weights, today_data, portfolio, day, vol_scale=1.0,
    ) -> None:
        """Process target-weight rebalancing orders (Gap 15)."""
        prices = {}
        for symbol, row in today_data.items():
            close = row.get("Close")
            if close is not None and not pd.isna(close) and close > 0:
                prices[symbol] = close
        orders = portfolio.compute_rebalance_orders(target_weights, prices)
        for symbol, side, qty in orders:
            if qty <= 0:
                continue
            if vol_scale != 1.0 and side == Side.BUY:
                qty = int(qty * vol_scale)
                if qty <= 0:
                    continue
            order = Order(
                symbol=symbol,
                side=side,
                quantity=qty,
                order_type=OrderType.MARKET,
                signal_date=day,
            )
            self._broker.submit_order(order)

    def _check_delistings(self, day, today_data, portfolio) -> None:
        """Gap 2: Force-close positions for delisted symbols (absent >5 days)."""
        for symbol in list(portfolio.positions.keys()):
            if symbol in today_data:
                self._missing_days[symbol] = 0
            else:
                self._missing_days[symbol] = self._missing_days.get(symbol, 0) + 1
                if self._missing_days[symbol] > 5:
                    pos = portfolio.positions[symbol]
                    price = pos._market_price
                    if price > 0:
                        logger.warning(
                            f"Delisting detected: {symbol} absent {self._missing_days[symbol]} days. "
                            f"Force-closing at last known price ${price:.2f}"
                        )
                        if pos.total_quantity > 0:
                            trades = pos.sell_lots_fifo(pos.total_quantity, price, day, 0.0)
                            portfolio.trade_log.extend(trades)
                            portfolio.cash += price * sum(t.quantity for t in trades)
                        elif pos.total_quantity < 0:
                            cover_qty = abs(pos.total_quantity)
                            trades = pos.close_lots_fifo(cover_qty, price, day, 0.0)
                            portfolio.trade_log.extend(trades)
                            portfolio.cash -= price * cover_qty
                        portfolio.close_position(symbol)
                    del self._missing_days[symbol]

    def _process_drip(self, day, today_data, portfolio) -> None:
        """Gap 16: Reinvest dividends into additional shares."""
        for symbol, pos in list(portfolio.positions.items()):
            if pos.total_quantity <= 0:
                continue
            row = today_data.get(symbol)
            if row is None:
                continue
            div_amount = row.get("Dividends", 0.0)
            if pd.isna(div_amount) or div_amount <= 0:
                continue
            close_price = row.get("Close", 0)
            if close_price <= 0:
                continue
            total_div = div_amount * pos.total_quantity
            new_shares = int(total_div / close_price)
            if new_shares > 0:
                pos.add_lot(new_shares, close_price, day, 0.0)
                remainder = total_div - new_shares * close_price
                portfolio.cash += remainder
            else:
                portfolio.cash += total_div

    @staticmethod
    def _is_rebalance_day(day: date, prev_day: date) -> bool:
        """Gap 45: Check if today is a rebalance day based on schedule."""
        # Weekly: Monday or first trading day of the week
        if day.isocalendar()[1] != prev_day.isocalendar()[1]:
            return True
        return False

    def _is_rebalance_day_for_schedule(self, day: date, prev_day: date) -> bool:
        """Check rebalance schedule."""
        sched = self.config.rebalance_schedule
        if sched == "daily":
            return True
        elif sched == "weekly":
            return day.isocalendar()[1] != prev_day.isocalendar()[1]
        elif sched == "monthly":
            return day.month != prev_day.month
        elif sched == "quarterly":
            return (day.month - 1) // 3 != (prev_day.month - 1) // 3
        return True

    @staticmethod
    def _compute_vol_scale(portfolio: Portfolio, config) -> float:
        """Gap 20: Compute position size scaling factor for vol targeting."""
        equity_history = portfolio.equity_history
        lookback = config.portfolio_vol_lookback
        if len(equity_history) < lookback + 1:
            return 1.0
        values = [v for _, v in equity_history[-lookback - 1:]]
        returns = []
        for j in range(1, len(values)):
            if values[j - 1] > 0:
                returns.append(values[j] / values[j - 1] - 1.0)
        if len(returns) < 2:
            return 1.0
        import numpy as np
        realized_vol = np.std(returns) * math.sqrt(252)
        if realized_vol <= 0:
            return 1.0
        scale = config.target_portfolio_vol / realized_vol
        return min(scale, 2.0)  # cap at 200%

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
        elif rf.condition == "fast_below_slow":
            return fast < slow

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
