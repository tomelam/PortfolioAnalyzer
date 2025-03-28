# Test Plan: Settings and CLI Behavior

This checklist helps verify the behavior of portfolio analysis logic across TOML files, config files, and CLI combinations.

---

## ✅ Portfolio TOML–based Settings

- [ ] Benchmark file, name, and date format override via TOML
- [ ] Risk-free rate file override in TOML
- [ ] Drawdown threshold set via TOML (and used correctly as a fraction)
- [ ] TOML file with missing optional fields — test fallback defaults
- [ ] TOML file with invalid fields or typos — should not crash

---

## ✅ Config File–based Settings

- [ ] Use of a custom config file: `--config other.toml`
- [ ] Config file that sets:
  - [ ] `output_dir`
  - [ ] `debug`
  - [ ] `do_not_plot`
  - [ ] `output_csv`
  - [ ] `output_snapshot`
  - [ ] `drawdown_threshold`
- [ ] Config file with no CLI overrides — confirm defaults
- [ ] Config file with invalid fields — should not crash

---

## ✅ Command-Line Override Handling

- [ ] CLI options override config values
- [ ] CLI works if config is missing
- [ ] `--save-golden-data` works as expected
- [ ] `--do-not-plot` suppresses plotting to screen and disk
- [ ] `--output-snapshot` + `--save-output-to` saves images to correct directory
- [ ] `--output-snapshot` without `--save-output-to` uses default output dir
- [ ] `--debug` enables traceback and disables quiet mode in age checks

---

## ✅ Staleness and Safety Behaviors

- [ ] Benchmark and risk-free files are checked for freshness
- [ ] Age warning prompt appears interactively
- [ ] `--skip-age-check` suppresses the prompt
- [ ] `quiet=True` suppresses prompt output in automation mode
- [ ] Program exits cleanly on "n" response to prompt

---

## ✅ Failure & Fallback Behavior

- [ ] Benchmark file not found — clear error
- [ ] Risk-free rate file not found — clear error
- [ ] Invalid benchmark file format — clean failure
- [ ] Incorrect date format — logs a warning, doesn’t crash
- [ ] Prompt fails silently in non-interactive shell unless skipped
