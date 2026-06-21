# Data refresh / freshness

PortfolioAnalyzer reads its benchmark and risk-free series from CSVs in
`data/`. These are kept current automatically — no more manual investing.com
downloads, and no cron. The program is self-sufficient: a normal run refreshes
what is behind and defends correctness itself.

## Why freshness is enforced, not advised

Stale **reference** data silently corrupts results: a stale benchmark skews
alpha/beta, a stale risk-free rate skews Sharpe/Sortino/alpha. So freshness is
treated as a **correctness invariant**, not a user-tunable policy. Mutual-fund
NAVs are fetched live every run (current by construction); PPF is a manual CSV
with no feed to pull (it can only warn). The auto-refreshable reference feeds
are:

| Data | Local file | Upstream | Cadence |
|---|---|---|---|
| Risk-free rate | `data/reference/INDIRLTLT01STM.csv` | FRED `INDIRLTLT01STM` (India 10Y govt bond rate) | monthly |
| Benchmark (NIFTY 50 TRI) | `data/reference/NIFTY Total Returns Historical Data.csv` | niftyindices.com Total-Returns API | business day |
| Gold price | `data/reference/gold_lbma_usd_daily.csv` | LBMA Gold Price PM fix (USD/troy-ounce), `prices.lbma.org.uk` | business day |
| USD/INR FX | `data/reference/DEXINUS.csv` | FRED `DEXINUS` (Indian Rupees per US Dollar) | business day |

The gold feed is gated **only for portfolios that hold gold or SGBs** (both are
valued off it); the FX feed is gated **only for portfolios that hold SGBs** (an
INR instrument whose rupee coupons are converted to USD — see below). Other
portfolios never touch either. All are no-auth feeds. FRED and
LBMA serve clean machine-readable responses (CSV / JSON) over plain `requests`.
niftyindices is aggressively
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
   day** (NIFTY, gold) or **month** (FRED). There is no age-tolerance knob.
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
4. **Provenance travels with the output.** For each reference source the run
   reports `last_date` (latest data point), `fetched_at` (when our copy was
   pulled), and `attempted_at` (last refresh attempt), from
   `data/.last_fetched.json`. It is echoed to **stderr**, drawn as a footnote
   on the PNG, and embedded in the PNG `tEXt` metadata (with a human-readable
   metrics block — see `docs/OUTPUTS.md`).
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

## Gold is an auto-refreshed source (USD, daily)

Gold **used to be** a manual, feedless CSV (monthly INR averages from the World
Gold Council). That dataset was **discontinued in March 2025** when ICE
Benchmark Administration pulled the historical LBMA Gold Price from third-party
redistributors — so the old "refresh by hand" path is dead, not merely neglected.

Gold now refreshes automatically from the **LBMA's own** price feed
(`prices.lbma.org.uk/json/gold_pm.json`) — the canonical London auction
benchmark, the freest fair gold market — as the **PM fix in USD per troy
ounce**, daily, back to 1968, over plain `requests` (no auth, key, or browser).
It is treated exactly like the benchmark/risk-free feeds: a gold/SGB-bearing run
refreshes it when behind the business-day cadence and **blocks** if it can't be
certified current (override with `--allow-stale`). Non-gold portfolios never
touch it. `--as-of`/`--replay-from` runs neither fetch nor block (data pinned).

The price is kept in **USD, unconverted**: the analyzer reports only normalized
returns, and a time-varying USD/INR path would inject rupee/RBI dynamics into
gold's measured return (a constant oz→gram unit cancels under normalization; a
varying FX path does not). The per-gram conversion (`/31.1034768`) still holds,
now USD/gram.

## SGBs add a USD/INR FX dependency (coupons only)

An SGB is an **INR** instrument, but there is no free, deep-history *daily* INR
gold series, so the holding is marked in **USD** off the LBMA gold spot above.
Its 2.5% coupon is contractually a **rupee** cash amount, so the SGB engine
(`sgb_holdings.sgb_holding_civ`) converts each coupon to USD at **its own
payment date's** USD/INR rate (FRED `DEXINUS`, INR per USD: `usd = inr /
DEXINUS`) and the CIV is a single consistent USD series. FX touches only the
discrete rupee coupon amounts — **never the gold price path** (the sanctioned
"contractual cash value" exception; a constant unit cancels, a varying FX path
on the price would not).

This is the honest decoupling of gold and SGB: a **gold-only** run gates on the
LBMA feed alone; an **SGB** run gates on LBMA gold **and** the `DEXINUS` FX
feed. The contractual **INR** premature/maturity redemption *view* (RBI/IBJA),
shown co-equally with the HTM mark, is still to come; its **data layer** now
exists (next section), but it does not yet feed a valuation (see
`docs/ARCHITECTURE.md`).

## SGB redemption prices — event-cadence ingest (RBI press releases)

The contractual redemption value of an SGB is fixed by the **RBI**, not by LBMA
gold: ~3 business days before each redemption date RBI publishes a press release
setting the price (₹ per unit; 1 unit = 1 gram of 999 gold) as the simple
average of the IBJA closing gold price over the preceding three business days.
`loaders/sgb_redemptions.py` ingests these announcements into
`data/funds/sgb_redemptions.csv` (`tranche_id, redemption_date, kind ∈
{PRE,MAT}, inr_per_gram, source_prid_or_url, tier`).

This is **not** one of the cadence-gated reference feeds above: redemptions land
on irregular scheduled dates (an *event* cadence, not daily/monthly), so there
is no calendar frontier to measure "behind" against, and — until the redemption
*view* valuation is wired — no metric depends on it, so nothing blocks on it. It
is a growing multi-column **table**, not a single-value time series, so it has
its own loader rather than a `data_update` `DataSource` (it reuses that module's
stamp machinery for unified provenance in `data/.last_fetched.json`).

Source path (established by `scripts/probe_rbi_sgb_redemption.py`, the B2
Step-0 gate): the mobile directory `BS_SwarnaBharat.aspx` is CAPTCHA-walled, so
we route around it over plain `requests` — enumerate redemption press releases
via RBI's open search endpoint (`SearchResults.aspx`), then parse each
`BS_PressReleaseDisplay.aspx?prid=<N>` for the date, ₹/unit price, and
tranche(s). No IBJA paid API or stealth browser needed. Refresh on demand:

```bash
./venv/bin/python -m portfolioanalyzer.loaders.sgb_redemptions
```

The committed CSV is seeded with the redemptions RBI's search currently lists
(most recent ~14, all tier **T1** — fetched direct from the press release).
Older premature redemptions and maturity (MAT) redemptions are documented gaps
to backfill; a re-run upserts new announcements on `(tranche_id,
redemption_date)` and never drops existing rows.

## Manual, feedless sources (PPF)

PPF has **no upstream feed** to auto-refresh, so it can't be enforced like the
reference data — the program can only *warn*, never block or fetch (see
`docs/ARCHITECTURE.md` → *Data freshness as a correctness invariant*). You
maintain it by hand.

- **PPF — `data/funds/ppf_interest_rates.csv`** (declared rate by effective date).
  **Sparse by design**: one row per rate *change*, not per month. The loader
  carries the latest rate forward, so a months-old last row usually just means
  the rate is unchanged (7.1% since 2019) — **not** staleness, which is why PPF
  is deliberately *not* cadence-warned (a monthly check would false-alarm
  forever). The review cadence is **quarterly**: when the government announces a
  PPF rate change, add a `YYYY-MM-DD,rate` row. The loader **fails fast** on an
  unparseable date or non-numeric rate (a silently-dropped row would corrupt the
  CIV — this caught a real `2025=01-01` typo).

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
