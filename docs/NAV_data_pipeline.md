### STAGE 0: INPUTS

──────────────────────────────────────

- CSVs for each mutual fund
- Settings / portfolio file

          ↓
### STAGE 1: LOAD + COMBINE

──────────────────────────────────────

Function: `fetch_portfolio_civs(portfolio)`

Returns:  dict of DataFrames  →  {"Fund A": dfA, "Fund B": dfB, ...}

          ↓

Function: `align_portfolio_civs(...)`

Output:   one aligned DataFrame → `aligned_portfolio_civs`

         ✅ GOOD place to use a DataFrame:
         - Easier to align multiple NAVs by date
         - Easier to drop rows with missing data
         - Easy to visualize or debug all funds at once

### STAGE 3: CONVERSION FOR PORTFOLIO MODEL

──────────────────────────────────────

Custom logic in `main.py`:

    fund_series_dict = {
        fund_name: aligned_portfolio_civs[fund_name]
        for fund_name in aligned_portfolio_civs.columns
    }

         ❗ Transition point:
         - Convert each column (Series) into separate dict
           entry
         - Required for OOP modeling

          ↓
### STAGE 4: BUILD OBJECTS

──────────────────────────────────────

Function: `from_multiple_nav_series(nav_dict)`

Input:    {"Fund A": seriesA, "Fund B": seriesB, ...}

Output:   `PortfolioTimeseries` object with per-fund `AssetTimeseries`

         ✅ Each asset becomes a standalone unit
         - Supports weight assignment
         - Individual metrics
         - Future extensions like drawdown, alpha, Sharpe...

         ❌ A DataFrame would get in the way here
         - Would require ugly slicing and assumptions

### STAGE 5: CALCULATE METRICS

──────────────────────────────────────

Method: `portfolio.combined_daily_returns()`
Output: Daily returns series for portfolio as a whole

Feeds into:
- Risk adjustment
- Performance metrics
- Visualization

         ✅ Return to using single Series or DataFrame as
            appropriate
         - But now it's clean, purposeful, and structured
