# 📜 PortfolioAnalyzer: Class Roles and Responsibilities

## 1. `AssetTimeseriesManager`
- **Purpose:** Manage single asset's timeseries
- **Input:** Cumulative Investment Value (**CIV**) or NAV series (pd.Series).
- **Handles:**
  - Normalizes the NAV/CIV series.
  - Calculates asset-specific returns, alpha, beta, volatility, etc.
- **Data type:** **NAV/CIV series** (converts internally to returns when needed).

---

## 2. `PortfolioTimeseriesManager`
- **Purpose:** Manage collection of assets
- **Input:** Dictionary of assets and weights (assets are `AssetTimeseriesManager` instances).
- **Handles:**
  - Alignment of asset data.
  - Weighted aggregation of returns or NAVs.
  - Produces portfolio-wide daily or monthly returns.
- **Data type:** **NAV/CIV series** (aggregates into a combined NAV series, then returns).

---

## 3. `TimeseriesNav`
- **Purpose:** Transform NAVs to returns
- **Input:** NAV/CIV series (pd.Series).
- **Handles:**
  - Resampling monthly or using daily NAVs.
  - Produces clean monthly or daily returns.
- **Data type:** **NAV/CIV series** ➔ **Returns series**.

---

## 4. `TimeseriesReturn`
- **Purpose:** Analyze clean returns series
- **Input:** Clean returns series (pd.Series), not NAVs.
- **Handles:**
  - Calculation of metrics (Sharpe, Sortino, volatility, CAGR, max drawdown, etc.).
  - No resampling; assumes returns are ready.
- **Data type:** **Return series** (already differenced, not cumulative NAVs).

---

# 📣 Summary Table

| Class                      | Input Type    | Purpose                           |
|:---------------------------|:--------------|:----------------------------------|
| `AssetTimeseriesManager`    | NAV/CIV Series | Manage single asset's timeseries  |
| `PortfolioTimeseriesManager`| NAV/CIV Series | Manage collection of assets       |
| `TimeseriesNav`             | NAV/CIV Series | Transform NAVs to returns         |
| `TimeseriesReturn`          | Returns Series | Analyze clean returns series      |

---

# 📜 Future Naming Plan (suggested)

| Old Name               | Suggested New Name            | Reason                                |
|:-----------------------|:-------------------------------|:--------------------------------------|
| `TimeseriesFrame`       | `TimeseriesReturn`             | Clearer, specifies it's returns only  |

---

✅ The Purpose lines are now sharp, compressed, and easy to scan quickly.

Would you also like an ultra-compact "Class Responsibility Map" (diagram)? (optional but cool!)

