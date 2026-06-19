# Full Global To-Do List for PortfolioAnalyzer

## Current Focus

1. Fix crash when running a one-mutual-fund portfolio
2. Fix broken unit tests after TimeseriesFrame → Timeseries structure
3. Rename TimeseriesFrame → Timeseries for clarity

## Future Improvements

4. Apply the previously suggested quick crash-stabilization patch for single-fund portfolios
5. Apply the full "surgical stabilization" audit (better single-fund handling after crash fixes)
6. Write minimal unit tests for Timeseries:
   - Constructor strictness
   - Auto-renaming input Series
   - `.index`, `.values`, `.iloc`, `.loc`, `.dropna` behavior
7. Remove old defensive assertions from `main.py`
8. Add developer documentation:
   - Table: when to use Timeseries, DataFrame, raw Series
9. (Optional) Create simple dataflow diagram (from CSVs to Timeseries to metrics)

---

# Key Working Principles

- **Fail Fast at Object Boundaries**: Constructor strictness over mainline assertions
- **Keep Main Program Narrative Clean**: Only real loading, crunching, saving
- **Separate Tables vs Timeseries Cleanly**: Tables (drawdowns, allocations) → DataFrames, Returns/NAVs → Timeseries
- **Progress Over Perfection**: Finish crash fixes before over-refactoring

