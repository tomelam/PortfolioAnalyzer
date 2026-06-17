# Data refresh / freshness

PortfolioAnalyzer reads its benchmark and risk-free series from CSVs in
`data/`. These are kept current automatically — no more manual investing.com
downloads, and no cron. The program is self-sufficient: a normal run refreshes
what is behind and defends correctness itself.

## Why freshness is enforced, not advised

Stale **reference** data silently corrupts results: a stale benchmark skews
alpha/beta, a stale risk-free rate skews Sharpe/Sortino/alpha. So freshness is
treated as a **correctness invariant**, not a user-tunable policy. Mutual-fund
NAVs are fetched live every run (current by construction); gold/PPF are manual
CSVs with no feed to pull (they can only warn). The two auto-refreshable
reference feeds are:

| Data | Local file | Upstream | Cadence |
|---|---|---|---|
| Risk-free rate | `data/INDIRLTLT01STM.csv` | FRED `INDIRLTLT01STM` (India 10Y govt bond rate) | monthly |
| Benchmark (NIFTY 50 TRI) | `data/NIFTY Total Returns Historical Data.csv` | niftyindices.com Total-Returns API | business day |

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

## How freshness works on every run

1. **"Behind" is the feed's own cadence, not a magic number.** A source is
   behind when its latest local date is older than the most recent **business
   day** (NIFTY) or **month** (FRED). There is no age-tolerance knob.
2. **Auto-refresh is baked in.** A source that is behind is refreshed before
   metrics are computed — refreshing *is* the remedy. A successful fetch yields
   the latest the source offers, so it is current by definition (even on an
   exchange holiday, where the calendar frontier may sit one day ahead of the
   last real data point — a harmless refresh that finds nothing new).
3. **Block by default.** If a reference source cannot be certified current
   (the upstream is unreachable, or — for niftyindices — already attempted
   today), the run **stops** rather than print degraded metrics. The single
   override is `--allow-stale`, which proceeds after printing a warning that
   names the affected metrics.
4. **Provenance is printed.** For each reference source the run reports both
   `last_date` (the latest data point — what bears on correctness) and
   `fetched_at` (when our copy was pulled), read from `data/.last_fetched.json`.
5. **Deterministic modes opt out cleanly.** `--as-of DATE` and `--replay-from`
   neither fetch nor block: data pinned through the as-of date is current as of
   that date.

## niftyindices is contacted at most once per day

niftyindices throttles **bursts** very aggressively and a block is held against
the source IP, so it is the one genuine guard in the system. Every *attempt*
(success or failure) is stamped in `data/.last_fetched.json`, and a second
attempt the same day is suppressed — if a refresh failed earlier today, the run
won't poke the host again; it treats the benchmark as stale (and blocks unless
`--allow-stale`). FRED carries no such risk, so it refreshes whenever it is
behind.

The fetch itself mirrors the proven `mysore-spa-intelligence-engine` scraper:
an authentic fingerprint (real UA, `en-IN` / `Asia/Kolkata`, 1920×1080, hides
`navigator.webdriver`), navigate the historical-data page to mint cookies /
clear any JS challenge, pause briefly, then issue the TRI POST from inside the
page so the request carries the session cookies. A single hit returns the full
date range — `fetch_niftyindices_tri` makes **one attempt, no retries** (rapid
retries only deepen a burst block; the "retry" is simply the next day's run).

Practical guidance:
- Install the `browser` extra (`pip install '.[browser]' && playwright install
  chromium`); without it the niftyindices fetch raises a clear, install-guiding
  error, which surfaces as a stale benchmark (refresh it manually, or run with
  `--allow-stale` if you accept degraded alpha/beta).
- A benchmark already current for the day triggers **zero** fetches.

## Manual one-shot refresh (optional)

`portfolio-analyzer-update` refreshes every registered source in one shot and
writes the `data/.last_fetched.json` stamp. It is **not** required — a normal
run already refreshes what it needs — but it is handy to warm the local CSVs
ahead of time (e.g. before an offline session):

```bash
./venv/bin/portfolio-analyzer-update
```

It exits non-zero only if *every* source failed (so one flaky feed doesn't fail
the command). There is intentionally no scheduled/cron path: the analyzer is
self-sufficient and refreshes on demand.

## Re-capturing goldens after a source change

The golden-master tests pin deterministic output (`--as-of 2026-06-13`,
`--replay-from tests/golden/replay`). If you change a benchmark/risk-free
**source** (not just refresh it), re-capture them — see
`docs/TESTING.md` → "Regenerating goldens".
