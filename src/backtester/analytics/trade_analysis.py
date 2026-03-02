"""Per-trade MAE/MFE analysis (Gap 21)."""

import pandas as pd


def compute_mae_mfe(
    trades: list,
    price_data: dict[str, pd.DataFrame],
) -> list[dict]:
    """Compute Maximum Adverse Excursion and Maximum Favorable Excursion per trade.

    Args:
        trades: List of Trade objects with entry_date, exit_date, entry_price, symbol.
        price_data: Dict of symbol -> OHLCV DataFrame.

    Returns:
        List of dicts with 'symbol', 'entry_date', 'exit_date', 'mae', 'mfe'.
    """
    results = []
    for trade in trades:
        df = price_data.get(trade.symbol)
        if df is None:
            continue

        entry_ts = pd.Timestamp(trade.entry_date)
        exit_ts = pd.Timestamp(trade.exit_date)

        mask = (df.index >= entry_ts) & (df.index <= exit_ts)
        holding = df.loc[mask]

        if holding.empty or trade.entry_price <= 0:
            results.append({
                "symbol": trade.symbol,
                "entry_date": trade.entry_date,
                "exit_date": trade.exit_date,
                "mae": 0.0,
                "mfe": 0.0,
            })
            continue

        entry = trade.entry_price

        if trade.quantity >= 0:
            # Long trade
            min_low = holding["Low"].min()
            max_high = holding["High"].max()
            mae = (min_low - entry) / entry
            mfe = (max_high - entry) / entry
        else:
            # Short trade
            max_high = holding["High"].max()
            min_low = holding["Low"].min()
            mae = (entry - max_high) / entry
            mfe = (entry - min_low) / entry

        results.append({
            "symbol": trade.symbol,
            "entry_date": trade.entry_date,
            "exit_date": trade.exit_date,
            "mae": mae,
            "mfe": mfe,
        })

    return results


def mae_mfe_summary(results: list[dict]) -> dict:
    """Compute summary statistics from MAE/MFE results."""
    if not results:
        return {"avg_mae": 0.0, "avg_mfe": 0.0, "efficiency": 0.0}

    maes = [r["mae"] for r in results]
    mfes = [r["mfe"] for r in results]

    avg_mae = sum(maes) / len(maes)
    avg_mfe = sum(mfes) / len(mfes)
    efficiency = abs(avg_mfe / avg_mae) if avg_mae != 0 else 0.0

    return {
        "avg_mae": avg_mae,
        "avg_mfe": avg_mfe,
        "efficiency": efficiency,
    }
