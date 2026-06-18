# Test Plan: Settings and CLI Behavior

This checklist tracks coverage of the settings/CLI surface across portfolio
TOMLs, the config file, and CLI flag combinations.

Note on terminology: benchmark / risk-free / drawdown / output settings are
**config-file** settings (the `--config` TOML), not portfolio-TOML fields. The
portfolio TOML holds `label` + `funds` (+ asset sleeves). The settings-merge
precedence (CLI > config-file > built-in default) lives in
`main.build_settings`, which `tests/unit/test_settings_merge.py` exercises
directly.

---

## ✅ Settings overridable via the config TOML

Covered by `tests/unit/test_settings_merge.py`.

- [x] Benchmark file, name, and date format override
- [x] Risk-free rate file (and risk-free date format) override
- [x] Drawdown threshold set via config (override now honored — previously
      shadowed by a baked-in argparse default; fixed alongside these tests).
      The fraction conversion (`/100`) is downstream in `main()`; not yet
      asserted end-to-end — see "remaining" below.
- [x] Missing optional fields → fall back to built-in defaults
- [x] Invalid/typo'd config keys → ignored, no crash

---

## ✅ Config-file mechanics

Covered by `tests/unit/test_settings_merge.py`.

- [x] Custom config file path is read (`--config other.toml`)
- [x] Config file sets: `output_dir`, `output_csv`, `output_snapshot`,
      `max_drawdown_threshold`, `metrics_method`, `allow_stale`, `quiet`,
      `lookback`, `show_plot`
- [x] Config file with no CLI overrides → built-in defaults confirmed
- [x] Missing config file → empty dict → complete default settings
- [x] Invalid/extra fields → no crash

Note: the legacy `do_not_plot` config key is now `show_plot` (CLI
`--disable-plot-display`); `debug` is a CLI store_true (`--debug`), not
typically a config key.

---

## ✅ Command-line override handling

Covered by `tests/unit/test_settings_merge.py` unless noted.

- [x] CLI options override config values (output dir, csv, metrics method,
      drawdown threshold, allow-stale, quiet, lookback, show-plot)
- [x] CLI works if config is missing
- [x] `--output-dir` absent → default `outputs/` (was: `--save-output-to`,
      since renamed to `--output-dir`)
- [x] `--as-of` parsed to a normalized Timestamp; absent → None
- [x] `--disable-plot-display` → `show_plot=False` (was: `--do-not-plot`)
- ~~`--save-golden-data`~~ — RETIRED. Replaced by `--save-replay` /
  `--replay-from` (golden capture/replay); see `test_golden_master.py`.
- ~~`--debug` … disables quiet mode in age checks~~ — RETIRED with the
  freshness redesign (no more age-check prompts). `--debug` still enables
  tracebacks (exercised implicitly by the e2e failure tests).

---

## ✅ Data-freshness invariant (block-by-default)

Covered by `tests/unit/test_data_update.py` (cadence logic / once-a-day gate)
and `tests/unit/test_freshness_gate.py` (the main.py block / --allow-stale glue).

- [x] Reference sources are refreshed when behind their publication cadence
- [x] A source that can't be certified current blocks the run by default
- [x] `--allow-stale` proceeds with a warning naming the degraded metrics
- [x] niftyindices is contacted at most once per day (attempt-stamped)
- [x] `--as-of` / `--replay-from` neither fetch nor block (deterministic)
- [x] Retired flags (`--skip-age-check`/`--no-auto-update`/`--max-riskfree-delay`)
      fail fast with a pointer to `--allow-stale`
- [x] Goldens are insulated from the on-run data refresh (reference inputs
      frozen under `tests/golden/replay/reference/`); enforced by
      `test_golden_master.test_golden_config_does_not_read_live_data_dir`

---

## ✅ Failure & fallback behavior

- [x] Benchmark file not found → clear error, non-zero exit
      (`tests/integration/test_main_e2e.py`)
- [x] Risk-free rate file not found → clear error, non-zero exit
      (`tests/integration/test_main_e2e.py`)
- [x] Missing portfolio TOML → non-zero exit
      (`tests/integration/test_main_e2e.py`)
- [x] Invalid `--as-of` string → fails loudly, not silently
      (`tests/unit/test_settings_merge.py`)

---

## Remaining (not yet covered)

- [ ] Invalid benchmark file *format* (well-formed path, garbage contents) →
      clean failure with a useful message
- [ ] Incorrect benchmark/risk-free date format → warning, not a crash
- [ ] Drawdown threshold fraction conversion (`/100`) asserted end-to-end,
      not just at the settings-merge layer
