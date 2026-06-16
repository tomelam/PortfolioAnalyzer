| Variable                        | Prefix         | Stem            | Suffix  | Type                       | Content                                                                   |
|---------------------------------|----------------|-----------------|---------|----------------------------|---------------------------------------------------------------------------|
| `benchmark_returns_series`      | —              | benchmark       | returns | `pd.Series`                | Daily-gain series for the chosen benchmark index                          |
| `aligned_portfolio_civs`        | aligned        | portfolio_civs  | —       | `pd.DataFrame`             | CIV time-series for each fund, aligned on common dates                    |
| `unaligned_portfolio_civs`      | unaligned      | portfolio_civs  | —       | `dict[str, pd.Series]`     | Raw CIV series per fund before alignment                                  |
| `nav_inputs`                    | —              | nav_inputs      | —       | `dict[str, pd.Series]`     | NAV series inputs (equity funds, bonds, gold) for building `portfolio_ts` |
| `portfolio_ts`                  | —              | portfolio_ts    | —       | `PortfolioTimeseries`      | Object encapsulating combined NAV/CIV series and return series methods    |
| `gain_daily_portfolio_series`   | gain_daily     | portfolio       | series  | `pd.Series`                | Daily portfolio returns from `combined_daily_returns()`                   |
| `portfolio_civ_series`          | portfolio      | civ             | series  | `pd.Series`                | Cumulative Investment Value series from `combined_civ_series()`           |
| `aligned_risk_free_rate_series` | —              | risk_free_rates | —       | `pd.Series`                | Risk‐free rates aligned to portfolio dates                                |
| `risk_free_rate`                | risk_free_rate | —               | —       | `float`                    | Mean of `aligned_risk_free_rate_series`                                   |
| `risk_free_rate_daily`          | risk_free_rate | daily           | —       | `float`                    | Daily equivalent of the annualized risk‐free rate                         |
| `portfolio_returns`             | portfolio      | returns         | —       | `TimeseriesReturn`         | Wrapper for the portfolio’s return series                                 |
| `benchmark_returns`             | benchmark      | returns         | —       | `TimeseriesReturn`         | Wrapper for the benchmark’s return series                                 |
| `cumulative_historical`         | cumulative     | historical      | —       | `np.ndarray` / `pd.Series` | Cumulative portfolio returns for plotting                                 |
| `cumulative_benchmark`          | cumulative     | benchmark       | —       | `np.ndarray` / `pd.Series` | Cumulative benchmark returns for plotting                                 |
