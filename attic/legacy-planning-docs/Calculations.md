# PortfolioAnalyzer Documentation Enhancements

## PortfolioAnalyzer Strengths

- **Calculates essential metrics with high rigor:** (CAGR, volatility, max drawdown, recovery time).
- **Uses daily NAV returns** rather than monthly snapshots to accurately capture real-world risk.
- **Fail-fast, fail-loud design:** Immediately surfaces bugs instead of hiding them.
- **Finance-correct definitions:** Sharpe, Sortino, Alpha, Beta suitable for rigorous backtesting.
- **Robust handling of missing data:** Ensures data hygiene and accuracy.
- **Transparent calculations:** Fully traceable and understandable metrics.

### Bragging Sentence for README

> **PortfolioAnalyzer delivers professional-grade backtesting metrics based on daily returns, robust risk handling, and transparent calculations—outperforming typical retail tools in accuracy and realism.**

## Golden Spec Sheet

**Program Assumptions for Portfolio Metrics:**

- **Return Type:** Daily total return (NAV-based).
- **Risk-Free Rate:** India 10-Year Government Bond Yields.
- **Risk-Free Compounding:** Daily simple rate (annual rate divided by 365).
- **Annualization Base:** 252 trading days per year.
- **Benchmark:** Explicitly specified per portfolio (e.g., NIFTY TRI).
- **Sharpe Ratio:** Mean Excess Return divided by Standard Deviation of Returns.
- **Sortino Ratio:** Mean Excess Return divided by Standard Deviation of Downside Returns.
    - **Downside Threshold:** Risk-free rate.
    - **Fuzziness Threshold:** -1e-10 (tolerance for tiny negative noise).
- **Alpha, Beta:** Based on CAPM regression against benchmark daily returns.
- **Drawdowns:** Based on peak-to-trough declines in CIV series.

## Daily vs Monthly Data Impact Cheat Sheet

PortfolioAnalyzer calculates metrics based on **daily returns**, providing more accurate risk and drawdown estimates than platforms using monthly data:

| Metric        | Daily vs Monthly Data Impact |
|---------------|------------------------------|
| CAGR          | Tiny difference |
| Volatility    | Monthly returns understate true volatility |
| Sharpe Ratio  | Monthly returns slightly inflate Sharpe |
| Sortino Ratio | Monthly returns noticeably inflate Sortino |
| Max Drawdown  | Monthly data misses worst intramonth drops |
| Alpha, Beta   | More stable and realistic with daily data |

> **Daily-based metrics are stricter and more realistic. Use daily returns for serious analysis. Use monthly returns only for simple summaries.**

## Optional Polish Checklist (Future Improvements)

- Allow flexible override of risk-free rate.
- Let user select annualization settings (252 vs 365 days).
- Allow explicit specification of benchmark source type (price index vs total return).
- Provide user options for Alpha calculation (regression vs arithmetic).
- Allow user-defined Sortino downside threshold.
- Implement rolling metrics (1-year Sharpe, Sortino, Alpha).
- Enhance drawdown recovery calculations for partial recoveries.
- Integrate golden reference datasets for validation.
- Allow precision control of metrics display.
- Develop high-fidelity unit tests for edge-case scenarios.

