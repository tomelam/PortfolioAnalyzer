# data/reference/ — reference time series

Static/reference market & rate series. Two groups:

## Wired — loaded by code or a tracked config

| File | What | Coverage | Loaded by |
|---|---|---|---|
| `NIFTY Total Returns Historical Data.csv` | NIFTY 50 **TRI** (total return) | 2007 → 2025-05 | `main.py` `benchmark_returns_file` default; goldens |
| `INDIRLTLT01STM.csv` | FRED India **long-term** govt bond rate (monthly) | 2011-12 → | `main.py` `risk_free_rates_file` default |
| `gold_lbma_usd_daily.csv` | LBMA Gold Price PM fix, **USD/troy-ounce** (daily) | 1968 → | `main.py` `gold_prices_file` default; `loaders/gold.py`; **auto-refreshed** (`loaders/data_update.py` `gold_lbma`) |
| `DEXINUS.csv` | FRED USD/INR (**Indian Rupees per US Dollar**, daily) | 1973 → | `main.py` `fx_usd_inr_file` default; SGB coupon→USD conversion (`sgb_holdings.py`); **auto-refreshed** (`loaders/data_update.py` `fx_usd_inr`) |
| `India 10-Year Bond Yield Historical Data.csv` | India 10Y yield (investing.com export) | — | `examples/config/mid-cap_config.toml`, `Makefile` |
| `Nifty Midcap 150 Historical Data.csv` | NIFTY Midcap 150 (investing.com export) | — | `examples/config/mid-cap_config.toml` |
| `rbi_91day_tbills_from_dbie.csv` | RBI 91-day T-bill yield (DBIE) | 1993 → | `examples/config/mid-cap_config.toml` |

## Unwired candidates — salvaged, not yet referenced by any loader/config

These are genuine data, just never wired in. Candidate benchmark / risk-free inputs.
**Caveat: the NIFTY index dumps are Price-Return (PR), not Total-Return (TR)** — don't
treat them as TRI without adjustment.

| File | What | Coverage | Note |
|---|---|---|---|
| `INDIR3TIB01STM.csv` | FRED India **3-month** T-bill rate (monthly) | 2011-12 → 2025-02 | Distinct **short-rate** series; complements the long-term `INDIRLTLT01STM.csv`. |
| `NIFTY MIDCAP 50_Historical_PR_*.csv` | NIFTY Midcap 50 (PR) | 2007 → 2025 | niftyindices export. |
| `NIFTY MIDCAP 100_Historical_PR_*.csv` | NIFTY Midcap 100 (PR) | 2007 → 2025 | niftyindices export. |
| `NIFTY MIDCAP 150_Historical_PR_01012005to24032025.csv` | NIFTY Midcap 150 (PR) | 2005 → 2025 | Longest-coverage of the 3 snapshots that existed; the 2 redundant 2007-start snapshots are in `attic/superseded-snapshots/`. |
| `NIFTY SMALLCAP 50_Historical_PR_*.csv` | NIFTY Smallcap 50 (PR) | 2007 → 2025 | niftyindices export. |
| `NIFTY SMALLCAP 100_Historical_PR_*.csv` | NIFTY Smallcap 100 (PR) | 2007 → 2025 | niftyindices export. |
| `NIFTY SMALLCAP 250_Historical_PR_*.csv` | NIFTY Smallcap 250 (PR) | 2007 → 2025 | niftyindices export. |
| `NIFTY NEXT 50_Historical_PR_*.csv` | NIFTY Next 50 (PR) | 2007 → 2025 | niftyindices export. |
| `NIFTY TOTAL MARKET_Historical_PR_*.csv` | NIFTY Total Market (PR) | 2007 → 2025 | niftyindices export. |
