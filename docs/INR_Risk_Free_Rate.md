## INR Risk-Free Rate Handling

PortfolioAnalyzer supports use of 91-day Indian T-Bill YTMs as a proxy for the INR risk-free rate. Two common approaches:

### 1. Daily Conversion + Annualization

Convert each YTM (expressed as a % per 91-day holding period) into a daily equivalent:

```
daily_rf = (1 + yield_percent / 100) ** (1 / 252) - 1
```

Then annualize for Sharpe and CAPM alpha calculations:

```
annual_rf = (1 + daily_rf) ** 252 - 1
```

### 2. Monthly Smoothing

Interpolate the YTMs to monthly frequency using a method like linear or spline interpolation. Then convert the interpolated yields to monthly-equivalent returns:

```
monthly_rf = (1 + yield_percent / 100) ** (1 / 12) - 1
```

**Note:** Choose one approach based on your metrics method (`--metrics-method daily|monthly`). Ensure the data is recent and consistently formatted.
