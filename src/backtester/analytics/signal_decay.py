"""Signal decay analysis: measure how trade entry signals perform over time.

For each trade's entry, compute cumulative returns at T+1, T+2, ..., T+N days
to understand signal strength decay and identify optimal holding periods.
"""

import numpy as np
import pandas as pd


def compute_signal_returns(trades, price_data: dict[str, pd.DataFrame],
                           max_horizon: int = 20) -> pd.DataFrame:
    """Compute cumulative returns at each horizon for every trade entry signal.

    For each trade, uses the Close price on the entry_date as the base, then
    calculates cumulative return at T+1, T+2, ..., T+max_horizon using
    subsequent Close prices from the ticker's DataFrame index (trading days).

    Post-exit periods are marked NaN to avoid contaminating the analysis with
    returns that the strategy could not have captured.

    Args:
        trades: List of Trade objects (must have entry_date, exit_date,
                entry_price, symbol attributes).
        price_data: Dict of ticker -> OHLCV DataFrame with DatetimeIndex.
        max_horizon: Maximum number of trading days to look ahead.

    Returns:
        DataFrame with columns: trade_id, ticker, entry_date, signal_side,
        T+1, T+2, ..., T+{max_horizon}. Each T+N column is the cumulative
        return from entry Close to that day's Close.
    """
    if not trades:
        cols = ["trade_id", "ticker", "entry_date", "signal_side"]
        cols += [f"T+{h}" for h in range(1, max_horizon + 1)]
        return pd.DataFrame(columns=cols)

    horizon_cols = [f"T+{h}" for h in range(1, max_horizon + 1)]
    rows = []

    for idx, trade in enumerate(trades):
        ticker = trade.symbol
        df = price_data.get(ticker)
        if df is None:
            # Ticker not in price_data -- fill all horizons with NaN
            row = {
                "trade_id": idx,
                "ticker": ticker,
                "entry_date": trade.entry_date,
                "signal_side": "BUY",
            }
            for col in horizon_cols:
                row[col] = np.nan
            rows.append(row)
            continue

        # Find the entry date's position in the DataFrame index.
        # Convert entry_date to Timestamp for consistent comparison.
        entry_ts = pd.Timestamp(trade.entry_date)
        if entry_ts not in df.index:
            # Entry date not found in data -- fill with NaN
            row = {
                "trade_id": idx,
                "ticker": ticker,
                "entry_date": trade.entry_date,
                "signal_side": "BUY",
            }
            for col in horizon_cols:
                row[col] = np.nan
            rows.append(row)
            continue

        entry_loc = df.index.get_loc(entry_ts)
        entry_close = df["Close"].iloc[entry_loc]

        # Convert exit_date for comparison
        exit_ts = pd.Timestamp(trade.exit_date)

        row = {
            "trade_id": idx,
            "ticker": ticker,
            "entry_date": trade.entry_date,
            "signal_side": "BUY",
        }

        for h in range(1, max_horizon + 1):
            future_loc = entry_loc + h
            if future_loc >= len(df):
                row[f"T+{h}"] = np.nan
                continue

            future_ts = df.index[future_loc]

            # If the trade was already exited before this date, mark NaN
            if future_ts > exit_ts:
                row[f"T+{h}"] = np.nan
                continue

            future_close = df["Close"].iloc[future_loc]
            row[f"T+{h}"] = (future_close - entry_close) / entry_close

        rows.append(row)

    return pd.DataFrame(rows, columns=["trade_id", "ticker", "entry_date",
                                        "signal_side"] + horizon_cols)


def average_signal_decay(signal_returns_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Aggregate per-trade signal returns into mean, median, and std at each horizon.

    Args:
        signal_returns_df: DataFrame from compute_signal_returns().

    Returns:
        Tuple of (mean_series, median_series, std_series), each indexed by
        horizon labels (T+1, T+2, ..., T+N).
    """
    horizon_cols = [c for c in signal_returns_df.columns if c.startswith("T+")]

    if signal_returns_df.empty or not horizon_cols:
        empty = pd.Series(dtype=float)
        return empty, empty.copy(), empty.copy()

    data = signal_returns_df[horizon_cols]
    mean_series = data.mean()
    median_series = data.median()
    std_series = data.std()

    return mean_series, median_series, std_series


def optimal_holding_period(signal_returns_df: pd.DataFrame) -> dict:
    """Find the horizon where average cumulative return peaks.

    Args:
        signal_returns_df: DataFrame from compute_signal_returns().

    Returns:
        Dict with:
        - optimal_days: horizon (int) where mean return is highest
        - peak_return: the mean return at that horizon
        - return_at_max_horizon: mean return at the last horizon
    """
    mean_series, _, _ = average_signal_decay(signal_returns_df)

    if mean_series.empty:
        return {
            "optimal_days": 0,
            "peak_return": 0.0,
            "return_at_max_horizon": 0.0,
        }

    # Extract the integer horizon from column labels like "T+5"
    peak_label = mean_series.idxmax()
    peak_day = int(peak_label.split("+")[1])
    peak_return = float(mean_series[peak_label])
    return_at_max = float(mean_series.iloc[-1])

    # Check monotonic cases
    values = mean_series.dropna().values
    if len(values) >= 2:
        diffs = np.diff(values)
        if np.all(diffs >= 0):
            # Monotonically increasing -- optimal is max horizon
            peak_day = int(mean_series.index[-1].split("+")[1])
            peak_return = float(mean_series.iloc[-1])
        elif np.all(diffs <= 0):
            # Monotonically decreasing -- optimal is 1
            peak_day = 1
            peak_return = float(mean_series.iloc[0])

    return {
        "optimal_days": peak_day,
        "peak_return": peak_return,
        "return_at_max_horizon": return_at_max,
    }


def signal_decay_summary(trades, price_data: dict[str, pd.DataFrame],
                         max_horizon: int = 20) -> dict:
    """Convenience function: compute all signal decay analytics in one call.

    Args:
        trades: List of Trade objects.
        price_data: Dict of ticker -> OHLCV DataFrame.
        max_horizon: Maximum trading days to look ahead.

    Returns:
        Dict with:
        - per_trade_returns: DataFrame from compute_signal_returns
        - avg_decay: mean Series from average_signal_decay
        - median_decay: median Series
        - std_decay: std Series
        - optimal_holding: dict from optimal_holding_period
        - total_signals: count of trades analyzed
    """
    per_trade = compute_signal_returns(trades, price_data, max_horizon)
    mean_s, median_s, std_s = average_signal_decay(per_trade)
    optimal = optimal_holding_period(per_trade)

    return {
        "per_trade_returns": per_trade,
        "avg_decay": mean_s,
        "median_decay": median_s,
        "std_decay": std_s,
        "optimal_holding": optimal,
        "total_signals": len(trades),
    }
