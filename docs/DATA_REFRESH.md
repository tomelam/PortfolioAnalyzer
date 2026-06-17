# Data refresh / auto-update

PortfolioAnalyzer reads its benchmark and risk-free series from CSVs in
`data/`. These are kept current automatically — no more manual
investing.com downloads.

## Sources

| Data | Local file | Upstream | Cadence |
|---|---|---|---|
| Risk-free rate | `data/INDIRLTLT01STM.csv` | FRED `INDIRLTLT01STM` (India 10Y govt bond rate) | monthly |
| Benchmark (NIFTY 50 TRI) | `data/NIFTY Total Returns Historical Data.csv` | niftyindices.com Total-Returns API | daily |

Both are stable, no-auth feeds. FRED serves a clean CSV. niftyindices is
fetched through its session-cookie + JSON-POST API (`loaders/data_update.py`),
which is the robust way past its anti-scrape wall.

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

niftyindices throttles **bursts** very aggressively: a single isolated request
succeeds (verified — it returns full TRI history in one call), but a handful in
a short window trips an IP block that takes minutes to clear. Because of that,
`fetch_niftyindices_tri` makes a **single attempt by default** (`retries=1`) —
rapid in-process retries can't beat the block and only deepen it. The real
"retry" is the next scheduled run.

Practical guidance:
- Prefer the **daily cron** (`portfolio-analyzer-update`) — one hit/day never
  trips the limit.
- A fresh local file triggers **zero** requests (refresh only fires when stale).
- Don't run the live network test in a loop; if throttled it skips (clearly
  labelled upstream throttling, not a code defect), and an on-run refresh just
  warns and uses existing data.

## Re-capturing goldens after a source change

The golden-master tests pin deterministic output (`--as-of 2026-06-13`,
`--replay-from tests/golden/replay`). If you change a benchmark/risk-free
**source** (not just refresh it), re-capture them — see
`docs/TESTING.md` → "Regenerating goldens".
