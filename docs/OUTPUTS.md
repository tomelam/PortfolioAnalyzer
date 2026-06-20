# Outputs policy: preserve by default

Every portfolio run writes its PNG snapshot and CSV(s) under `outputs/`.
These files are **expensive to recompute** — each run hits mfapi.in for
every fund's full NAV history, and the user often keeps historical
renders to compare against later changes (data refreshes, code changes,
new portfolios). The toolchain treats `outputs/` as a cache to grow,
never to silently shrink.

## Rules

1. **`make all`** — incremental. Rebuilds `outputs/<name>.png` and the
   sibling `outputs/<name>.csv` only when the target is missing or
   older than its `examples/port/<name>.toml`. Files already present are
   touched only if their TOML changed. No deletions ever happen as a
   side-effect of `make all`.

2. **`make rerender`** — force rebuild of every PNG + CSV regardless
   of mtime, but **without deleting** other files in `outputs/`. Use
   this when the code changed (e.g. a math fix) and every render
   needs to be regenerated against the same TOML set.

3. **`make clean`** — removes only the sweep-summary file
   `reports/portfolio_metrics.csv`. **Does not touch `outputs/`.**

4. **`make distclean`** — the only target that removes `outputs/`
   wholesale. Asks for `y` confirmation before doing so. Use when
   you're sure no historical render is worth keeping.

5. **`scripts/render-all.sh`** — equivalent to `make rerender` but
   pure bash (no Makefile dependency). Iterates every `examples/port/*.toml`
   and writes into `outputs/<name>/<name>.{png,csv,assets.csv,drawdowns.csv}`.
   Existing renders are overwritten in place; nothing else under
   `outputs/` is touched.

## Sibling files written alongside each portfolio

| File | Producer | Purpose |
|---|---|---|
| `<name>.png` | `visualizer.plot_cumulative_returns` | Cumulative-return plot + allocation table |
| `<name>.csv` | `main.py` | One-line metrics summary (CAGR, vol, Sharpe, …) |
| `<name>.drawdowns.csv` | `drawdowns_csv.write_drawdowns_csv` | One row per recovered drawdown |
| `<name>.assets.csv` | `fund_lifecycle.write_assets_csv` | Per-asset inauguration + DEFUNCT status |

The `.assets.csv` is the warning channel: when a fund has been silent
for more than 30 days its row reports `DEFUNCT` and `main.py` prints
the same warning on stdout. See `docs/ARCHITECTURE.md` for the
"effective window" mechanism that clips the portfolio CIV at the
earliest-dying asset's last NAV.

## Self-documenting PNG (embedded metadata)

Each `<name>.png` carries its run context **inside the file**, as PNG `tEXt`
chunks (written by `visualizer.plot_cumulative_returns` via
`output_metadata.build_png_metadata`). So an old snapshot is recoverable on its
own — no re-run, no matching CSV needed:

```bash
exiftool outputs/port-1.png            # shows the custom tags
python -c "from PIL import Image; print(Image.open('outputs/port-1.png').text['metrics'])"
```

Embedded keys: `portfolio`, `run` (`live` / `as-of YYYY-MM-DD` / `replay`),
`generated` (UTC), `metrics` (a labeled, human-readable block — far more usable
than the positional `<name>.csv` row), and `reference_data` (per-source
`last_date` / `fetched_at` / `attempted_at` for the benchmark + risk-free
feeds). The same reference-data provenance is also drawn as a small footnote on
the plot and echoed to stderr on every run. The metrics CSV schema is
unchanged — this is purely additive.

## Why this matters

Two failure modes the policy prevents:

- **Cheap accidents.** A misremembered `make clean` after weeks of
  iteration would otherwise delete months of comparison data the user
  cares about. The default-preserve policy makes that mistake impossible
  short of `make distclean` + explicit `y`.
- **Surprise refetches.** mfapi.in is fast but not free; refetching
  every fund's full history on every run pollutes the upstream and
  slows the dev loop. Incremental rebuild keeps refetches scoped to
  actually-changed portfolios.
