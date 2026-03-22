# Alternative Data Expansion — Implementation Plan

> Generated 2026-03-22 via multi-agent research, validation, and planning process.
> 10 features selected from 33 brainstormed ideas, validated by academic research,
> published literature, and practitioner anecdotes.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Selected Features](#2-selected-features)
3. [Architecture](#3-architecture)
4. [Implementation Phases](#4-implementation-phases)
5. [Phase A: EDGAR Extensions](#5-phase-a-edgar-extensions)
6. [Phase B: Market Data (VIX + Intermarket)](#6-phase-b-market-data)
7. [Phase C: FRED Integration](#7-phase-c-fred-integration)
8. [Phase D: External Data (Put-Call + Analyst)](#8-phase-d-external-data)
9. [Phase E: New Strategies](#9-phase-e-new-strategies)
10. [Phase F: CLI, Config, and Docs](#10-phase-f-cli-config-docs)
11. [Unit Test Plan](#11-unit-test-plan)
12. [E2E Test Plan](#12-e2e-test-plan)
13. [Documentation Updates](#13-documentation-updates)
14. [Risk Assessment](#14-risk-assessment)

---

## 1. Overview

### What We're Building

10 new alternative data capabilities for the backtesting engine, organized into 4 groups:

| Group | Features | New Dependencies | Effort |
|-------|----------|:----------------:|:------:|
| **EDGAR Extensions** | Piotroski F-Score, Altman Z-Score, Buyback/Shareholder Yield, Dividend Growth | None | Low |
| **Market Data** | VIX Term Structure, Cross-Asset Intermarket (Cu/Au) | None (yfinance) | Low-Med |
| **FRED Integration** | Macro Regime, Treasury Yield Curve | `fredapi` (optional) | Medium |
| **External Data** | CBOE Put-Call Ratio, Analyst Earnings Revisions | None | Medium |

### Why These Were Selected

Each feature passed a 3-validator gate (Phase 3):
- **Academic research** — peer-reviewed evidence of alpha (e.g., F-Score: 7-23%/yr, Z-Score: exclusion filter, Short interest: "strongest known aggregate predictor")
- **Published books** — coverage in major trading/investing literature
- **Practitioner anecdotes** — real-world usage by traders and institutions

### Key Architectural Principles

1. **All features are opt-in** — zero impact on existing behavior when disabled
2. **Graceful degradation** — strategies return HOLD when data columns absent (EDGAR pattern)
3. **Column prefix convention** — each data source gets a unique prefix (`fund_`, `vix_`, `intermarket_`, `fred_`, `yield_`, `sentiment_`, `analyst_`)
4. **Cache-first loading** — all external data cached to Parquet
5. **Optional dependency guarding** — `fredapi` guarded with `try/except ImportError` (like `edgartools`)
6. **No lookahead** — all data forward-filled, point-in-time safe

---

## 2. Selected Features

### Feature 1: Piotroski F-Score
- **What:** 9 binary financial health criteria summed to a 0-9 score
- **Academic backing:** 7-23%/yr alpha (Piotroski 2000, J. Accounting Research; replicated internationally by Hyde 2018)
- **Output column:** `fund_piotroski_f` (integer 0-9)
- **Data needed:** All fields already exist in TAG_MAP

### Feature 2: Altman Z-Score
- **What:** Weighted sum of 5 financial ratios predicting bankruptcy risk
- **Formula:** Z = 1.2(WC/TA) + 1.4(RE/TA) + 3.3(EBIT/TA) + 0.6(MVE/TL) + 1.0(Sales/TA)
- **Academic backing:** 72% accuracy predicting bankruptcy 2 years ahead (Altman 1968, JF)
- **Output columns:** `fund_altman_z` (float), `fund_altman_zone` (safe/grey/distress)
- **New TAG_MAP entries:** `retained_earnings` (RetainedEarningsAccumulatedDeficit), `total_liabilities` (Liabilities)

### Feature 3: Buyback/Shareholder Yield
- **What:** Net buyback yield + total shareholder yield (buybacks + dividends - dilution)
- **Academic backing:** 12-45% cumulative abnormal returns over 3-4 years (Ikenberry et al. 1995, JFE; Peyer & Vermaelen 2009, RFS)
- **Output columns:** `fund_buyback_yield`, `fund_shareholder_yield`
- **New TAG_MAP entries:** `stock_repurchased` (PaymentsForRepurchaseOfCommonStock), `stock_issued_proceeds` (ProceedsFromIssuanceOfCommonStock), `stock_comp` (ShareBasedCompensation)

### Feature 4: Dividend Growth
- **What:** YoY dividend growth rate, payout ratio, consecutive increase streak
- **Academic backing:** Dividend Aristocrats outperform with lower volatility (S&P research)
- **Output columns:** `fund_div_growth_yoy`, `fund_payout_ratio`
- **New TAG_MAP entries:** `dividends_per_share` (CommonStockDividendsPerShareDeclared)

### Feature 5: VIX Term Structure
- **What:** VIX/VIX3M contango-backwardation ratio as volatility regime signal
- **Academic backing:** Backwardation is statistically significant contrarian buy signal (Fassas 2019, JRFM)
- **Data source:** yfinance (^VIX, ^VIX3M, ^VIX9D, ^VVIX — already works)
- **Output columns:** `vix_close`, `vix_3m`, `vix_ratio`, `vix_regime`
- **Integration:** Engine-level regime filter (supplements SMA regime)

### Feature 6: Cross-Asset Intermarket
- **What:** Copper/gold ratio as macro risk-on/risk-off signal; dollar index
- **Academic backing:** Gold/copper ratio has robust predictive power (Roh et al. 2025, SSRN; Huang & Kilic 2019, JFE)
- **Data source:** yfinance (HG=F, GC=F, DX-Y.NYB, CL=F — already works)
- **Output columns:** `intermarket_cu_au_ratio`, `intermarket_cu_au_momentum`, `intermarket_dollar`

### Feature 7: FRED Macro Regime
- **What:** Composite regime score from yield curve slope, credit spreads, LEI, unemployment claims
- **Academic backing:** Combining multiple macro forecasts delivers ~2% OOS R² (Rapach et al. 2010, RFS)
- **Data source:** FRED API via `fredapi` (free, 120 req/min, free API key)
- **FRED series:** T10Y2Y, T10Y3M, BAMLH0A0HYM2, DBAA/DAAA, USSLIND, ICSA
- **Output columns:** `fred_yield_spread_10y2y`, `fred_credit_spread_hy`, `fred_macro_regime`
- **Integration:** Engine-level regime filter (supplements or replaces SMA regime)

### Feature 8: Treasury Yield Curve
- **What:** Full Treasury curve + TIPS breakevens + Fed Funds for rate-aware valuation
- **Academic backing:** "Financial gravity" — rate environment governs all asset valuation (Campbell 1987, JFE)
- **Data source:** FRED API (same dependency as Feature 7)
- **FRED series:** DGS1MO through DGS30, T5YIE, T10YIE, DFF
- **Output columns:** `yield_10y`, `yield_2y`, `yield_spread`, `yield_real_10y`

### Feature 9: CBOE Put-Call Ratio
- **What:** Daily equity put-call ratio as contrarian sentiment signal
- **Academic backing:** 40-50 bps/week at firm level (Pan & Poteshman 2006, RFS)
- **Data source:** CBOE free CSV download (no API key, no rate limits, back to 2003+)
- **Output columns:** `sentiment_pcr`, `sentiment_pcr_ma10`

### Feature 10: Analyst Earnings Estimate Revisions
- **What:** EPS revision direction, magnitude, breadth as momentum signal
- **Academic backing:** Tier-1 quant factor, persistent multi-month drift (Chan, Jegadeesh & Lakonishok 1996, JF)
- **Data source:** yfinance `eps_revisions`, `eps_trend`, `earnings_estimate` (already a dependency)
- **Output columns:** `analyst_rev_up_7d`, `analyst_rev_down_7d`, `analyst_rev_breadth`
- **Limitation:** yfinance provides current snapshots only, not historical point-in-time estimates

---

## 3. Architecture

### New Files to Create

```
src/backtester/data/
  market_data.py          — VIX term structure + cross-asset intermarket (Features 5 & 6)
  fred_source.py          — FRED API integration (Features 7 & 8)
  sentiment.py            — CBOE put-call ratio (Feature 9)
  analyst.py              — Analyst earnings revisions (Feature 10)

src/backtester/strategies/
  macro_aware_value.py    — Macro-regime + F-Score + valuation strategy
  sentiment_momentum.py   — Analyst revisions + insider + price momentum
  risk_regime.py          — VIX + yield curve + credit spread regime rotation
```

### Existing Files to Modify

| File | Changes |
|------|---------|
| `data/edgar_source.py` | Add 6 new TAG_MAP entries |
| `data/fundamental.py` | Add F-Score, Z-Score, buyback yield, dividend growth computations |
| `config.py` | Add `use_vix`, `use_intermarket`, `use_fred`, `use_pcr`, `use_analyst` fields |
| `cli.py` | Add CLI flags for each new data source |
| `engine.py` | Load/merge alternative data; enhance regime filter with VIX/FRED signals |
| `pyproject.toml` | Add `fred` optional dependency group |

### Data Flow (Updated Pipeline)

```
CLI flags (--use-vix, --use-fred, --use-pcr, etc.)
  └→ BacktestConfig (frozen)
       └→ BacktestEngine.run()
            ├→ DataManager.load_many()              — OHLCV for tickers
            ├→ EdgarDataManager.merge_all_onto_daily — fund_/insider_/inst_/event_ cols
            │    └→ NEW: fund_piotroski_f, fund_altman_z,
            │         fund_buyback_yield, fund_shareholder_yield,
            │         fund_div_growth_yoy, fund_payout_ratio
            ├→ NEW: MarketDataManager.load_vix()     — vix_ cols (once, merged to all)
            ├→ NEW: MarketDataManager.load_intermarket() — intermarket_ cols
            ├→ NEW: FredDataSource.load_macro_regime() — fred_ cols
            ├→ NEW: FredDataSource.load_yield_curve()  — yield_ cols
            ├→ NEW: CBOEPutCallSource.load()           — sentiment_ cols
            ├→ NEW: AnalystRevisionSource.fetch()      — analyst_ cols (per-symbol)
            ├→ strategy.compute_indicators()           — sees ALL new columns
            └→ Day loop:
                 ├→ _check_regime()  — NOW checks SMA AND/OR fred_macro_regime AND/OR vix_regime
                 └→ strategy.generate_signals() — can use any new column
```

### Column Prefix Convention

| Prefix | Source | Examples |
|--------|--------|----------|
| `fund_` | EDGAR financials (existing + new) | `fund_piotroski_f`, `fund_altman_z`, `fund_buyback_yield`, `fund_shareholder_yield`, `fund_div_growth_yoy`, `fund_payout_ratio` |
| `vix_` | VIX term structure | `vix_close`, `vix_3m`, `vix_ratio`, `vix_regime` |
| `intermarket_` | Cross-asset | `intermarket_cu_au_ratio`, `intermarket_cu_au_momentum`, `intermarket_dollar` |
| `fred_` | FRED macro | `fred_yield_spread_10y2y`, `fred_credit_spread_hy`, `fred_macro_regime` |
| `yield_` | Treasury curve | `yield_2y`, `yield_10y`, `yield_spread`, `yield_real_10y` |
| `sentiment_` | CBOE PCR | `sentiment_pcr`, `sentiment_pcr_ma10` |
| `analyst_` | Analyst revisions | `analyst_rev_up_7d`, `analyst_rev_down_7d`, `analyst_rev_breadth` |

### Cache Layout

```
{cache_dir}/
  market/VIX.parquet, VIX3M.parquet, HG_F.parquet, GC_F.parquet, ...
  fred/T10Y2Y.parquet, BAMLH0A0HYM2.parquet, DGS10.parquet, ...
  sentiment/equity_pc_ratio.parquet
  analyst/{SYMBOL}.parquet
  edgar/{type}/{SYMBOL}.parquet   (existing)
```

---

## 4. Implementation Phases

```
Phase A ──→ Phase B ──→ Phase C ──→ Phase D ──→ Phase E ──→ Phase F
(EDGAR)    (Market)    (FRED)     (External)  (Strategies) (CLI/Docs)
```

| Phase | Features | New Deps | Risk | Est. Scope |
|-------|----------|:--------:|:----:|:----------:|
| **A** | 1-4 (F-Score, Z-Score, Buyback, Dividends) | None | Low | Modify 2 files |
| **B** | 5-6 (VIX, Intermarket) | None | Low-Med | 1 new file + engine changes |
| **C** | 7-8 (FRED Macro, Yield Curve) | `fredapi` | Medium | 1 new file + engine regime filter |
| **D** | 9-10 (Put-Call, Analyst) | None | Medium | 2 new files |
| **E** | 3 new strategies | None | Low | 3 new files |
| **F** | CLI, config, docs | None | Low | Modify 4 files |

---

## 5. Phase A: EDGAR Extensions

### A1. TAG_MAP Additions (`edgar_source.py`)

Add 6 new entries to `TAG_MAP`:

```python
# Altman Z-Score
"retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
"total_liabilities": ["Liabilities"],

# Buyback / Shareholder Yield
"stock_repurchased": ["PaymentsForRepurchaseOfCommonStock"],
"stock_issued_proceeds": ["ProceedsFromIssuanceOfCommonStock"],
"stock_comp": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],

# Dividend Growth
"dividends_per_share": ["CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid"],
```

Update `_FLOW_METRICS` and `_STOCK_METRICS` sets accordingly:
- `_FLOW_METRICS`: add `stock_repurchased`, `stock_issued_proceeds`, `stock_comp`
- `_STOCK_METRICS`: add `retained_earnings`, `total_liabilities`, `dividends_per_share`

### A2. Piotroski F-Score (`fundamental.py`)

Add to `_compute_fundamental_ratios()`:

**9 binary criteria (each 0 or 1, summed to 0-9):**

| # | Criterion | Fields Used | Score 1 When |
|---|-----------|-------------|-------------|
| 1 | ROA | net_income / total_assets | ROA > 0 |
| 2 | CFO | operating_cf | CFO > 0 |
| 3 | ROA change | ROA vs ROA.shift(252) | ROA improved YoY |
| 4 | Accruals | operating_cf / total_assets vs ROA | CFO/TA > ROA |
| 5 | Leverage | total_debt / total_assets | Leverage decreased YoY |
| 6 | Liquidity | current_assets / current_liabilities | Current ratio improved YoY |
| 7 | Dilution | shares_outstanding | Shares unchanged or decreased YoY |
| 8 | Gross margin | gross_profit / revenue | Margin improved YoY |
| 9 | Asset turnover | revenue / total_assets | Turnover improved YoY |

Output: `fund_piotroski_f` (integer 0-9). NaN if insufficient data for YoY comparison.

### A3. Altman Z-Score (`fundamental.py`)

Add to `_compute_price_ratios()` (needs market cap from Close price):

```
Z = 1.2 * (working_capital / total_assets)
  + 1.4 * (retained_earnings / total_assets)
  + 3.3 * (operating_income / total_assets)
  + 0.6 * (market_cap / total_liabilities)
  + 1.0 * (revenue / total_assets)
```

Where: `working_capital = current_assets - current_liabilities`, `market_cap = Close * shares_outstanding`

Output: `fund_altman_z` (float). Zones: safe (>2.99), grey (1.8-2.99), distress (<1.8).

Division by zero: guard with `np.where(denominator != 0, ..., np.nan)`.

### A4. Buyback/Shareholder Yield (`fundamental.py`)

Add to `_compute_price_ratios()`:

```
market_cap = Close * fund_shares_outstanding
net_buyback = fund_stock_repurchased_ttm - fund_stock_issued_proceeds_ttm
fund_buyback_yield = net_buyback / market_cap
fund_shareholder_yield = (net_buyback + fund_dividends_paid_ttm - fund_stock_comp_ttm) / market_cap
```

### A5. Dividend Growth (`fundamental.py`)

Add new computation:

```
dps = fund_dividends_per_share
dps_prev = dps.shift(252)
fund_div_growth_yoy = (dps - dps_prev) / abs(dps_prev)
fund_payout_ratio = fund_dividends_paid_ttm / fund_net_income_ttm
```

---

## 6. Phase B: Market Data

### B1. New File: `data/market_data.py`

```python
class MarketDataManager:
    """Loads and caches VIX term structure + cross-asset intermarket data via yfinance."""

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir = cache_dir

    def load_vix_data(self, start: date, end: date) -> pd.DataFrame:
        """Load ^VIX, ^VIX3M via yfinance. Compute vix_ratio = VIX / VIX3M.
        vix_ratio > 1.0 = backwardation (stress), < 1.0 = contango (complacent).
        Cache to {cache_dir}/market/VIX.parquet etc."""

    def load_intermarket_data(self, start: date, end: date) -> pd.DataFrame:
        """Load HG=F (copper), GC=F (gold), DX-Y.NYB (dollar).
        Compute intermarket_cu_au_ratio = copper / gold.
        Compute intermarket_cu_au_momentum = ratio.pct_change(63).
        Cache similarly."""
```

### B2. Engine Integration (`engine.py`)

After EDGAR merge and before `compute_indicators()`:

```python
if config.use_vix or config.use_intermarket:
    from backtester.data.market_data import MarketDataManager
    mkt_mgr = MarketDataManager(cache_dir=config.data_cache_dir)

    if config.use_vix:
        vix_df = mkt_mgr.load_vix_data(config.start_date, config.end_date)
        for symbol in universe_data:
            universe_data[symbol] = _merge_auxiliary_data(universe_data[symbol], vix_df)
        # Also merge onto benchmark for regime filter
        benchmark_data = _merge_auxiliary_data(benchmark_data, vix_df)

    if config.use_intermarket:
        inter_df = mkt_mgr.load_intermarket_data(config.start_date, config.end_date)
        for symbol in universe_data:
            universe_data[symbol] = _merge_auxiliary_data(universe_data[symbol], inter_df)
```

Helper function:

```python
def _merge_auxiliary_data(daily_df: pd.DataFrame, aux_df: pd.DataFrame) -> pd.DataFrame:
    """Merge time-indexed auxiliary data onto daily DataFrame.
    Reindex to daily index + forward-fill (no lookahead)."""
    result = daily_df.copy()
    aux_reindexed = aux_df.reindex(result.index).ffill()
    for col in aux_reindexed.columns:
        result[col] = aux_reindexed[col]
    return result
```

### B3. VIX Regime Filter Enhancement (`engine.py`)

Modify `_check_regime()`:

```python
def _check_regime(self, benchmark_row):
    # ... existing SMA check ...
    sma_ok = True  # (existing logic)

    # NEW: VIX regime check (only if enabled)
    vix_ok = True
    if self.config.use_vix and benchmark_row is not None:
        vix_ratio = benchmark_row.get("vix_ratio")
        if vix_ratio is not None and not pd.isna(vix_ratio):
            vix_ok = vix_ratio < 1.0  # Contango = allow trades

    if self.config.fred_regime_mode == "replace":
        return fred_ok and vix_ok
    else:  # "supplement" — AND with SMA
        return sma_ok and vix_ok and fred_ok
```

---

## 7. Phase C: FRED Integration

### C1. New File: `data/fred_source.py`

```python
try:
    from fredapi import Fred
except ImportError:
    Fred = None

class FredDataSource:
    """Fetch macro data from FRED API. Optional dependency (like edgartools)."""

    MACRO_SERIES = {
        "fred_yield_spread_10y2y": "T10Y2Y",
        "fred_yield_spread_10y3m": "T10Y3M",
        "fred_credit_spread_hy": "BAMLH0A0HYM2",
        "fred_credit_spread_baa": "DBAA",  # subtract DAAA for BAA-AAA
        "fred_credit_spread_aaa": "DAAA",
        "fred_lei": "USSLIND",
        "fred_claims": "ICSA",
    }

    YIELD_SERIES = {
        "yield_3m": "DGS3MO",
        "yield_2y": "DGS2",
        "yield_5y": "DGS5",
        "yield_10y": "DGS10",
        "yield_30y": "DGS30",
        "yield_breakeven_5y": "T5YIE",
        "yield_breakeven_10y": "T10YIE",
        "yield_fed_funds": "DFF",
    }

    def __init__(self, api_key=None, cache_dir=None):
        key = api_key or os.environ.get("FRED_API_KEY")
        if Fred is None:
            raise ImportError("fredapi required: pip install -e '.[fred]'")
        self._fred = Fred(api_key=key)
        self._cache_dir = Path(cache_dir) / "fred" if cache_dir else None

    def load_macro_regime(self, start, end) -> pd.DataFrame:
        """Load macro series, compute composite regime score.
        Score = count of bullish conditions / total conditions.
        Bullish when: yield_spread > 0, credit_spread < median,
        LEI rising, claims falling."""

    def load_yield_curve(self, start, end) -> pd.DataFrame:
        """Load full Treasury curve + breakevens + Fed Funds.
        Compute yield_spread = 10Y - 2Y, yield_real_10y = 10Y - breakeven."""
```

### C2. FRED Regime Filter (`engine.py`)

Add to `_check_regime()`:

```python
fred_ok = True
if self.config.use_fred and benchmark_row is not None:
    macro_score = benchmark_row.get("fred_macro_regime")
    if macro_score is not None and not pd.isna(macro_score):
        fred_ok = macro_score >= 0.5  # Majority of indicators bullish
```

### C3. `pyproject.toml`

```toml
[project.optional-dependencies]
edgar = ["edgartools"]
fred = ["fredapi>=0.5"]
all = ["edgartools", "fredapi>=0.5"]
```

---

## 8. Phase D: External Data

### D1. CBOE Put-Call Ratio — New File: `data/sentiment.py`

```python
class CBOEPutCallSource:
    """Download CBOE equity put-call ratio from free CSV endpoint."""

    EQUITY_PC_URL = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv"

    def __init__(self, cache_dir=None):
        self._cache_dir = Path(cache_dir) / "sentiment" if cache_dir else None

    def load(self, start, end) -> pd.DataFrame:
        """Download CSV, parse, compute rolling averages.
        Columns: sentiment_pcr, sentiment_pcr_ma10.
        Falls back to cached data on download failure."""
```

### D2. Analyst Revisions — New File: `data/analyst.py`

```python
class AnalystRevisionSource:
    """Analyst earnings revisions via yfinance (per-symbol)."""

    def fetch(self, symbol: str) -> pd.DataFrame:
        """Get current eps_revisions, eps_trend from yfinance.
        Columns: analyst_rev_up_7d, analyst_rev_down_7d, analyst_rev_breadth.

        LIMITATION: yfinance provides current snapshot only, not historical.
        For backtesting, this means columns will be NaN for historical dates.
        Most useful for forward-looking / paper trading."""
```

---

## 9. Phase E: New Strategies

### E1. `macro_aware_value.py` — Macro-Regime + F-Score + Valuation

Adjusts quality thresholds based on macro regime:
- Expansion (positive yield curve, tight credit): accept F-Score >= 5, P/E < 20
- Contraction (inverted curve, wide spreads): require F-Score >= 7, P/E < 15
- Always excludes distressed stocks (Z-Score < 1.8)
- Uses: `fund_piotroski_f`, `fund_altman_z`, `fund_pe_ratio`, `fred_yield_spread_10y2y`, `fred_credit_spread_hy`

### E2. `sentiment_momentum.py` — Analyst Revisions + Insider + Momentum

Combines multiple sentiment/information signals:
- BUY when 2+ of: analysts revising UP, insiders buying, price above SMA
- SELL when 2+ of: analysts revising DOWN, insiders selling, price below SMA
- Uses: `analyst_rev_breadth`, `insider_buy_ratio_90d`, SMA indicator

### E3. `risk_regime.py` — VIX + Yield Curve + Credit Regime Rotation

Macro regime-driven allocation:
- Classifies market into risk-on/neutral/cautious/risk-off based on VIX term structure + yield curve + credit spreads
- Risk-on: favor growth/momentum stocks
- Risk-off: sell all or rotate to quality/value
- Uses: `vix_ratio`, `fred_yield_spread_10y2y`, `fred_credit_spread_hy`, `fund_piotroski_f`

---

## 10. Phase F: CLI, Config, and Docs

### Config Additions (`config.py`)

```python
@dataclass(frozen=True)
class BacktestConfig:
    # ... existing fields ...
    use_vix: bool = False
    use_intermarket: bool = False
    use_fred: bool = False
    fred_api_key: str | None = None
    fred_regime_mode: str = "supplement"  # "supplement" or "replace"
    use_yield_curve: bool = False
    use_pcr: bool = False
    use_analyst: bool = False
```

### CLI Additions (`cli.py`)

```
--use-vix               Enable VIX term structure data
--use-intermarket       Enable cross-asset intermarket signals
--use-fred              Enable FRED macro data
--fred-api-key TEXT     FRED API key (or set FRED_API_KEY env var)
--fred-regime-mode      {supplement, replace} — how FRED regime interacts with SMA regime
--use-yield-curve       Enable Treasury yield curve data
--use-pcr               Enable CBOE put-call ratio
--use-analyst           Enable analyst earnings revisions
```

All flags default to disabled. Existing behavior is unchanged.

### Example Commands

```bash
# EDGAR derived metrics (no new flags needed — just --use-edgar)
backtester run --strategy macro_aware_value --tickers AAPL MSFT GOOG \
  --benchmark SPY --start 2015-01-01 --end 2023-12-31 --use-edgar --cash 100000

# VIX regime filter
backtester run --strategy sma_crossover --tickers SPY QQQ --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 --use-vix

# FRED macro regime
export FRED_API_KEY=your_key_here
backtester run --strategy macro_aware_value --tickers AAPL MSFT --benchmark SPY \
  --start 2010-01-01 --end 2023-12-31 --use-edgar --use-fred

# Full alternative data stack
backtester run --strategy risk_regime --tickers SPY QQQ IWM --benchmark SPY \
  --start 2015-01-01 --end 2023-12-31 \
  --use-edgar --use-fred --use-vix --use-intermarket --use-pcr --cash 100000
```

---

## 11. Unit Test Plan

**~178 tests across 10 new test files. All synthetic data, zero network calls.**

| Test File | Feature(s) | Tests | Key Edge Cases |
|-----------|-----------|:-----:|----------------|
| `test_piotroski.py` | F-Score | 18 | Perfect 9, worst 0, each criterion individually, NaN in one metric, zero total_assets |
| `test_altman_zscore.py` | Z-Score | 16 | Safe/grey/distress zones, boundaries at 1.8 and 3.0, zero total_assets, zero total_liabilities |
| `test_shareholder_yield.py` | Buyback + Shareholder Yield | 15 | Net positive/negative buyback, issuance dilution, NaN dividend, TTM aggregation, point-in-time |
| `test_dividend_growth.py` | Dividend Growth | 16 | YoY positive/negative/zero, streak counting, broken streak, single year, flat dividend |
| `test_vix_term_structure.py` | VIX Term Structure | 18 | Contango/backwardation/flat, zero VIX, NaN handling, regime classification boundaries |
| `test_intermarket.py` | Cross-Asset | 17 | Cu/Au ratio, momentum lookback, zero gold, risk-on/off signals, alignment |
| `test_macro_regime.py` | FRED Macro | 20 | CSV parsing, composite score range, partial NaN, publication lag, regime transitions |
| `test_yield_curve.py` | Treasury Curve | 19 | Spread positive/negative/zero, curve classification, rate-aware PE thresholds, zero rate |
| `test_put_call_ratio.py` | CBOE PCR | 18 | CSV parsing, moving averages, extreme detection, boundary thresholds |
| `test_earnings_revisions.py` | Analyst Revisions | 21 | Revision breadth (all up/all down/mixed), trend direction, surprise %, zero consensus |

### Cross-Cutting Test Concerns

Every test file verifies:
1. Correct computation with hand-verified known inputs/outputs
2. NaN / missing data graceful degradation (never crashes)
3. Division by zero protection (returns NaN)
4. Empty DataFrame handling (returns NaN columns, no exceptions)
5. Correct column creation (prefix, dtype)
6. No lookahead (data only visible after publication date where applicable)
7. Boundary conditions at classification thresholds

---

## 12. E2E Test Plan

**59 tests in `tests/test_phase5_e2e.py`. Full BacktestEngine.run() pipeline.**

| Test Class | Feature | Tests | What's Mocked |
|------------|---------|:-----:|---------------|
| `TestPiotroskiFScoreE2E` | F-Score | 5 | DataSource only |
| `TestAltmanZScoreE2E` | Z-Score | 5 | DataSource only |
| `TestBuybackYieldE2E` | Buyback | 5 | DataSource only |
| `TestDividendGrowthE2E` | Dividends | 5 | DataSource only |
| `TestVIXTermStructureE2E` | VIX regime | 6 | DataSource + VIX tickers |
| `TestIntermarketE2E` | Intermarket | 5 | DataSource + commodity tickers |
| `TestFREDMacroRegimeE2E` | FRED regime | 6 | DataSource + FRED API |
| `TestTreasuryYieldCurveE2E` | Yield curve | 5 | DataSource + FRED API |
| `TestPutCallRatioE2E` | PCR | 6 | DataSource + CBOE CSV |
| `TestAnalystRevisionsE2E` | Analyst | 6 | DataSource + yfinance analyst |
| `TestCrossFeatureCompositionE2E` | Combined | 5 | All |

### Key E2E Scenarios Per Feature

Each feature tests:
1. **Feature enabled, data present** — signals generated, orders execute
2. **Feature enabled, data absent** — graceful degradation, HOLD signals, no crash
3. **Feature disabled** — zero impact on existing behavior
4. **Cross-feature composition** — multiple features active simultaneously
5. **Invariants** — no lookahead (T+1 fills), cash accounting, equity positive

### Standard E2E Invariants

```python
# Checked in every E2E test:
assert result.equity_series is not None
assert all(v > 0 for v in result.equity_series.values())  # equity always positive
# T+1 fill check:
for trade in result.trades:
    assert trade.fill_date > trade.signal_date
# Cash accounting after force-close:
assert abs(portfolio.cash - portfolio.total_equity) < 0.01
```

---

## 13. Documentation Updates

### README.md

- Add "Alternative Data Sources" section listing all 10 features
- Add `pip install -e ".[fred]"` installation instruction
- Add 4 new CLI examples (VIX regime, FRED macro, intermarket, EDGAR derived)
- Add new CLI options to the options table
- Update architecture tree with new files
- Update test count

### CLAUDE.md (project root)

- Update Language & Tooling with `fredapi` optional dep
- Add new files to Architecture tree
- Add new Design Rules: alternative data flow, FRED caching, VIX regime filter
- Add new test files to Test Files section

### .claude/CLAUDE.md

- Update Module Map with Alternative Data row
- Update Key Data Flows with alt data merge step
- Add new Critical Invariants: FRED rate limiting, alt data graceful degradation, column prefixes, cache isolation
- Update Patterns with optional-dependency guard, derived metrics, env var fallback
- Update test counts

### pyproject.toml

```toml
[project.optional-dependencies]
edgar = ["edgartools"]
fred = ["fredapi>=0.5"]
all = ["edgartools", "fredapi>=0.5"]
```

---

## 14. Risk Assessment

### Low Risk
- **Features 1-4 (EDGAR extensions):** Pure computation on existing data. Well-defined formulas. Only risk is XBRL tag availability (some companies may not report all fields). Mitigated by NaN propagation.

### Medium Risk
- **Features 5-6 (VIX, Intermarket):** Relies on yfinance for futures tickers which can be flaky. Mitigated by graceful degradation + Parquet caching + CBOE CSV fallback for VIX.
- **Feature 9 (CBOE PCR):** External CSV download; URL may change. Mitigated by cache fallback.

### Higher Risk
- **Feature 7 (FRED Macro Regime):** Modifies the regime filter — a critical engine component. A bug here suppresses/allows trades across all strategies. Mitigated by:
  1. `fred_regime_mode="supplement"` default (AND with existing SMA filter)
  2. When `use_fred=False` (default), zero code paths change
  3. Thorough E2E tests with both modes
- **Feature 10 (Analyst Revisions):** yfinance provides current snapshots only, not historical point-in-time estimates. This is a fundamental lookahead problem for backtesting. Mitigated by:
  1. Documenting the limitation clearly
  2. Columns will be NaN for historical dates → strategies HOLD
  3. Feature is most useful for forward-looking / paper trading

### Edge Cases to Handle

| Edge Case | Mitigation |
|-----------|------------|
| Date alignment (FRED/CBOE not on NYSE calendar) | Forward-fill via `reindex().ffill()` |
| Missing data (any new column NaN) | Strategies check with `pd.isna()` → HOLD |
| Backward compatibility | All features off by default; existing tests pass unchanged |
| Cache isolation | Separate subdirectories per data type |
| Optional dependency (`fredapi`) | `try/except ImportError` guard (like edgartools) |
| API key management (FRED) | CLI flag + env var fallback |
| Rate limiting (FRED: 120/min) | Cache aggressively; 8 API calls max per backtest |

---

## Appendix: Research Summary

### Ideas Considered and Discarded (Phase 3)

33 ideas were brainstormed across 4 personas (Ideator, Veteran Quant, Retail Trader, Value Investor). 19 were discarded during validation:

- **Reddit/Social Media** — seminal Twitter paper failed replication; signal degraded post-2021
- **Google Trends** — look-ahead bias in data; pytrends archived April 2025
- **Congressional Trading** — no good free source; alpha eliminated post-STOCK Act
- **Bitcoin as Barometer** — contemporaneous not leading; unstable correlation
- **Weather/NOAA** — statistically significant but economically marginal for equities
- **Spotify Mood** — Spotify deprecated audio features API Nov 2024
- **Yelp/Google Reviews** — no free historical data ($229+/month minimum)
- **Wikipedia Page Views** — short-term effect reverses; net zero
- **GitHub Activity** — zero empirical validation
- **Lunar Cycles** — data mining artifact despite published papers
- **Sports Outcomes** — real but fires too infrequently
- **Celebrity Death Index** — no academic support, untradeable
- And others (see Phase 3 validation reports for full details)

### Ideas That Passed But Were Not Selected

- **Fama-French Factor Momentum** — strong research but implementation challenges; factor timing often hurts in practice
- **Short Interest / DTC** — strongest known aggregate predictor but only ~1 year rolling from FINRA free tier
- **Industrial Electricity Consumption** — Sharpe Award winner (R²=9%) but monthly frequency limits utility
