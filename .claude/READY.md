# Task Execution Plan

## Independent Tasks (can run in parallel)

These tasks touch non-overlapping files and have no ordering constraints between them:

| Task | Files | Rationale |
|------|-------|-----------|
| **01 - Deduplicate metrics return alignment** | `analytics/metrics.py` | Pure refactor of one leaf module. No other task touches metrics.py. |
| **03 - Add calendar analytics tests** | `tests/test_calendar_analytics.py` (new) | New test file only. Reads `analytics/calendar.py` but does not modify it. |

Tasks 01 and 03 can run simultaneously with no risk of conflict.

## Sequenced Tasks

### Sequence A: CLI refactoring chain

**05 (deduplicate CLI config) -> 02 (add CLI tests) -> 08 (strategy registration)**

1. **05 first** -- Consolidates `_build_config` in `cli.py`. Must land before writing CLI tests so the tests validate the cleaned-up code, not the duplicated version.
2. **02 second** -- Adds `tests/test_cli.py`. Writing tests against the already-refactored CLI avoids rewriting test expectations later.
3. **08 last** -- Changes strategy registration in `cli.py` (replaces `# noqa: F401` imports with `discover_strategies()`), `conftest.py`, and `registry.py`. Should come after CLI tests exist (task 02) so regressions from the registration change are caught. Also modifies `conftest.py`, which affects all tests.

### Sequence B: Engine extraction chain

**04 (extract BacktestResult) -> 06 (extract stop logic)**

1. **04 first** -- Moves `BacktestResult` out of `engine.py` into `result.py`. This reduces engine.py and changes its import surface.
2. **06 second** -- Extracts ~115 lines of stop logic from `engine.py` into `execution/stops.py`. Doing this after task 04 avoids merge conflicts in engine.py (both tasks delete/move substantial blocks from the same file). Task 06 also needs the engine in a stable state to correctly identify the stop methods to extract.

### Sequence C: Report tests after BacktestResult extraction

**04 -> 07 (add report output tests)**

- **07** creates `tests/test_report.py` and must construct `BacktestResult` objects. If task 04 has already moved `BacktestResult` to `result.py`, the test imports should target the new location. Running 07 after 04 avoids writing tests with stale import paths.

## Conflicting Assumptions and Overlapping File Scope

### engine.py (HIGH conflict risk)
- **Task 04** removes `BacktestResult` class (~27 lines) and adds an import.
- **Task 06** removes ~115 lines of stop logic and adds `StopManager` calls.
- Both tasks perform large deletions in the same file. **Must be serialized** (04 then 06, or vice versa). Recommended order: 04 first (smaller, simpler change).

### cli.py (MODERATE conflict risk)
- **Task 05** refactors `_build_config` and the `run` command body.
- **Task 08** replaces strategy import lines and adds `discover_strategies()` call.
- **Task 02** may need minor refactoring to inject `DataManager` for testing.
- All three touch `cli.py`. **Serialize as 05 -> 02 -> 08.**

### conftest.py (LOW conflict risk)
- **Task 08** replaces `# noqa: F401` strategy imports with `discover_strategies()`.
- No other task modifies `conftest.py`, but all test tasks depend on it working correctly. **Run 08 last** in its sequence.

### analytics/report.py (LOW conflict risk)
- **Task 04** changes its import of `BacktestResult`.
- **Task 07** writes tests against it.
- No actual code conflict, but **07 should follow 04** so test imports match the new module layout.

## Recommended Execution Order

```
  Parallel Group 1:        01, 03
  Parallel Group 2:        04, 05
  Parallel Group 3:        06, 02, 07    (06 after 04; 02 after 05; 07 after 04)
  Final:                   08            (after 02)
```

Total: 4 rounds if maximizing parallelism.
