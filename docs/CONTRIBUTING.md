# Contributing

## TDD-first

Every behavior change starts with a failing test. The cycle:

1. **Red** — write the test against the *desired* contract; confirm it fails for the right reason.
2. **Green** — minimal change to pass.
3. **Refactor** — clean up, keeping the test green.
4. **Regression** — `pytest -m ""` (full suite incl. goldens) must stay green; if a golden moves, the commit message must explain *why*.

This isn't process for its own sake. The two Phase D CIV bugs were each
caught by a TDD test written *before* the fix — without that, the bug would
have been "explained away" by the existing goldens (which were themselves
wrong because they captured buggy output).

## Decomposition

Prefer pure functions in dedicated modules over methods on big classes.
`metrics.py` is the model: each metric is a function on a pandas Series, no
hidden state, trivially testable. Class methods (`TimeseriesReturn.sharpe`)
delegate.

## Pre-commit

Configured hooks (`.pre-commit-config.yaml`):

- `ruff` (lint + format)
- `mypy` (warn-only, will tighten module-by-module)
- whitespace / EOF / TOML / large-file checks

```bash
pre-commit install
pre-commit run --all-files
```

## KANBAN

`KANBAN.md` at repo root is the single source of truth for project maturity.
**Update it in the same commit as the code/test change**, not after the fact.
Move items between Backlog / In Progress / Done; don't delete history.

The plan that drove the salvage is at
`~/.claude/plans/concurrent-stargazing-shore.md`. Phase F is the final stretch
before the `v0.1-salvage` tag.

## Branch / commit conventions

- Branch names: `refactor/<topic>`, `fix/<topic>`, `test/<topic>`, `docs/<topic>`.
- Commits: imperative-mood subject, short. Body explains *why*, not *what* (`git diff` already shows what).
- One logical change per commit. If a refactor and a bug fix land in the same diff, split them.

## Don't

- Don't regenerate goldens to "make the test pass." If a golden moves and you
  haven't deliberately changed math, that's a regression to investigate.
- Don't add yfinance or similar unofficial-scraper data sources. They break
  silently when the upstream HTML changes. Prefer documented public APIs or
  static CSVs the user refreshes manually.
- Don't add abstractions for hypothetical future requirements. Three similar
  lines beats a premature base class.
- Don't add docstrings or comments that describe *what* the code does — the
  code already does that. Use comments for *why* (a hidden invariant, a
  workaround for a specific bug, a non-obvious constraint).

## Running the app

```bash
./venv/bin/python main.py \
  --config tests/fixtures/golden_master_config.toml \
  --metrics-method monthly \
  --lookback 5Y \
  port/port-1.toml
```

`tests/fixtures/golden_master_config.toml` is the test-runner config (skips
age checks against the stale data files); for real interactive runs use
`config/example_config.toml` and supply fresh data.
