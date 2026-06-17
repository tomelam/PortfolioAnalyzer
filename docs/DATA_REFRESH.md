# Data refresh / auto-update

PortfolioAnalyzer reads its benchmark and risk-free series from CSVs in
`data/`. These are kept current automatically — no more manual
investing.com downloads.

## Sources

| Data | Local file | Upstream | Cadence |
|---|---|---|---|
| Risk-free rate | `data/INDIRLTLT01STM.csv` | FRED `INDIRLTLT01STM` (India 10Y govt bond rate) | monthly |
| Benchmark (NIFTY 50 TRI) | `data/NIFTY Total Returns Historical Data.csv` | niftyindices.com Total-Returns API | daily |

Both are no-auth feeds. FRED serves a clean CSV. niftyindices is aggressively
anti-scrape and holds blocks against the source IP, so it is fetched **only**
through a stealth Chromium browser (`loaders/data_update.py`), never raw
`requests`. This needs the optional `browser` extra:

```bash
pip install '.[browser]' && playwright install chromium
```

The risk-free default switched from the manual investing.com 10Y CSV to the
FRED series so it can be auto-refreshed. The economic series is the same
(India 10Y govt yield); only Sharpe/Sortino/Alpha shift slightly. To keep the
old source, set `risk_free_rates_file` / `riskfree_date_format` in your config.

## How refresh happens

**On every run (default).** A normal `portfolio-analyzer` run force-refreshes
any benchmark/risk-free file that is stale (benchmark > ~7 days,
risk-free > ~45 days). A reachable source is pulled fresh; an unreachable one
prints a warning and the run proceeds with the existing data (early warning,
not a hard block). Auto-update is **disabled** under:

- `--as-of YYYY-MM-DD` and `--replay-from DIR` (deterministic / offline modes)
- `--skip-age-check`
- `--no-auto-update` (explicit opt-out)
- `auto_update = false` in config

**Scheduled (recommended).** Run the cron-able updater so runs always read
fresh local files and never pay a fetch latency:

```bash
# daily at 06:00
0 6 * * *  cd /path/to/PortfolioAnalyzer && ./venv/bin/portfolio-analyzer-update
```

`portfolio-analyzer-update` refreshes every registered source, writes a
`data/.last_fetched.json` stamp, and exits non-zero only if *every* source
failed (so one flaky feed doesn't page you).

## niftyindices rate-limiting

niftyindices throttles **bursts** very aggressively and a block is held
against the source IP, so we never poke it with raw `requests`. The sole fetch
path is a **stealth Chromium browser** (mirroring the proven
`mysore-spa-intelligence-engine` scraper): it presents an authentic
fingerprint (real UA, `en-IN` / `Asia/Kolkata`, 1920×1080, hides
`navigator.webdriver`), navigates the historical-data page to mint cookies /
clear any JS challenge, pauses briefly, then issues the TRI POST from inside
the page so the request carries the session cookies. A single hit returns the
full date range — `fetch_niftyindices_tri` makes **one attempt, no retries**
(rapid retries only deepen a burst block). The real "retry" is the next
scheduled run. Verified live (single stealth hit returns real TRI data).

Practical guidance:
- Install the `browser` extra (`pip install '.[browser]' && playwright install
  chromium`); without it the niftyindices fetch raises a clear, install-guiding
  error and the rest of the update proceeds.
- Prefer the **daily cron** (`portfolio-analyzer-update`) — one hit/day never
  trips the limit.
- A fresh local file triggers **zero** fetches (refresh only fires when stale).
- Don't run the live network test in a loop; if throttled it skips (clearly
  labelled upstream throttling, not a code defect), and an on-run refresh just
  warns and uses existing data.

## Re-capturing goldens after a source change

The golden-master tests pin deterministic output (`--as-of 2026-06-13`,
`--replay-from tests/golden/replay`). If you change a benchmark/risk-free
**source** (not just refresh it), re-capture them — see
`docs/TESTING.md` → "Regenerating goldens".
