# Structure of the PortfolioAnalyzer

## The Problems

This structure was hammered out with ChatGPT's help. It is presented as a conversation between human and machine.

I would also like to clarify, via variable names, which variables represent CIV- or NAV-type value and which variables represent returns-type values, and which variables are daily returns and which are annualized. There is also the question of cumulative vs. incremental.

Where possible, it would be very good to isolate these factors each to its own module or subclass or something, because it is very hard to follow the program logic as it is. The variable names alone just do not make it clear, right now, what values the variables hold.

Create a perfect grand plan to address all my concerns:
1. variable names that are very long -- hard to remember, awkward to code
2. similar names, too many features/aspects of variables to encode in their names
3. variable names and near-same names floating around between modules rather than staying in just one tightly controlled, single-responsibility module or class
4. pieces of variable names that are obscure and hard to grok immediately

If it's not too much and won't distract you from these main concerns, also consider a plan to get me from here to there within 3-4 hours of us coding together. Consider that I just broke the program by deleting the function calculate_portfolio_metrics, but the unit tests work pretty well.

## The Machine's Answer

You’ve laid out the perfect storm of maintainability issues — and we can absolutely bring order to it.

Here’s the Grand Plan to tame the chaos:

### 🧠 High-Level Philosophy
* Don't overload names with all features (e.g. cumulative_daily_annualized_nav_return_normalized_series).

* Let structure (classes, modules) encode what names can't.

* Group logic and naming by what the data means, not just how it’s shaped.

* Keep transitions between types of data explicit and easy to track.

### ✅ Core Concepts to Separate
Each axis below deserves its own name or structural distinction:


| Concept               | Type                             | Examples (Clean Names)    |
|-----------------------|----------------------------------|---------------------------|
| Raw asset value       | NAVs, prices, bond unit values   | `nav_hdfc`, `price_gold`  |
| CIV                   | Harmonized, unified value stream | `civ_hdfc`, `civ_gold`    |
| Returns               | Daily gains (pct change)         | `ret_hdfc`, `ret_gold`    |
| Cumulative gain       | (1 + returns).cumprod()          | `cumret_hdfc`             |
| Annualized            | Metric or rate over >1 year      | `sharpe_annual`, `cagr`   |
| Volatility types      | Standard deviation of returns    | `vol_daily`, `vol_annual` |
| Alpha/Beta input      | Benchmark-aligned return series  | `ret_benchmark`           |

### 🧱 Suggested Structure (Modules/Classes)
1. `data_sources.py`

    Fetch & normalize raw NAVs, bond prices, etc.
Return only “clean” Series with dates and float values.

2. `civ_builder.py`

    Takes raw value series → constructs:

    * CIV series
    * Return series (`.pct_change()`)
    * Cumulative return series
    * Returns all 3 in a dataclass:

        ```@dataclass
        class AssetTimeseries:
            civ: pd.Series
            ret: pd.Series
            cumret: pd.Series
    Now names inside your program can be short: civ, ret, etc. because context comes from the object.

3. `metrics_calculator.py`

    Computes all metrics — only from clean AssetTimeseries.

    No crawling around for raw NAVs, no duplicated `.pct_change()`. Just:

        def sharpe(asset: AssetTimeseries, risk_free_rate: float): ...

4. `timeseries.py`

    This stays lean:
	
	    TimeseriesFrame class + .cagr(), .volatility(), etc.

### 🔤 Naming Conventions Summary

| Meaning         | Prefix/Suffix Suggestion                |
|---------------------|-------------------------------------|
| Raw NAV or price    | `nav_`, `price_`, `unit_`           |
| Unified value       | `civ_`                              |
| Daily returns       | `ret_`                              |
| Cumulative gains    | `cumret_`                           |
| Benchmark series    | `bench_`, `ret_bench_`              |
| Annualized          | `_annual` suffix                    |
| Raw vs clean        | Use class structure to disambiguate |

Avoid names like gain_daily_portfolio_series — too long and unclear.

### 🛠️ How to Get There in 3–4 Hours
#### Phase 1: Freeze the API for core metrics ✅ (30–45 min)
* Define AssetTimeseries class (or namedtuple for fast start).
* Implement one converter: from NAV → AssetTimeseries
* Fix main.py to use this for one mutual fund.
* Let metrics stay where they are for now.

#### Phase 2: Rename data consistently ✅ (30–45 min)
* Rename top-level variables using civ_, ret_, cumret_ prefixes
* Replace value_series, gain_daily_*, etc.
* Add comments when converting between types.

#### Phase 3: Refactor metrics_calculator.py to take AssetTimeseries ✅ (60 min)
* Replace ad hoc Series logic with .civ, .ret, .cumret
* Keep metrics modular and testable

#### Phase 4: Collapse redundant names and create a loader (60 min)
* Replace repeated rename("value") / .to_frame() with a loader function
* Test the entire pipeline from raw CSV/NAV → metrics summary

#### Optional: Add __repr__ and debug summaries to each class — huge for traceability.

### 📦 Final Result
* Clear, short names like civ, ret, cumret
* Clarity from structure, not from convoluted names
* Refactored main.py that is 5× easier to follow
* Strong base for future portfolio blending and plotting

Would you like me to drop in the starter code for AssetTimeseries, and a conversion function from a NAV Series into it?







