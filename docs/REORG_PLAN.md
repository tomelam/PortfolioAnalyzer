# Directory-tree reorganization — execution plan (for a fresh session)

> **STATUS — EXECUTED.** Stage 1 (data/ + config/ tidy) merged 2026-06-20. Stage 2
> (package move) executed 2026-06-20 on branch `reorg-package`: all code collapsed
> into one **flat** `portfolioanalyzer/` package (the flat layout was chosen over
> the subpackage grouping floated below in "Open"). This plan is retained as the
> historical move-map.

**Why:** the repo root is hard to scan — 15 flat top-level `.py` modules (+ the
`loaders/` and `timeseries/` packages), loose stray files, and a `data/` dir that
mixes live inputs, fund metadata, and ~13 orphaned index dumps. Goal: a tree a
human can scan in seconds and find any input/output/code file.

**Process:** start a **fresh session** (this one is long; a reorg is the highest
blast-radius change in the project and must not lose its move-map to a mid-flight
context summary). **Enter plan mode first**, produce the execution move-map, gate
the package-move stage behind explicit approval. Stage it into branches; full
**network-gated** suite + `--no-ff` merge per the branch/merge workflow; ASK before
each merge.

## Decided (this session, 2026-06-20)
- **No shims / no back-compat layer.** Anti-pattern here (no external consumers).
  Repoint the *real* things — `./pa`, console-script targets, docs — at the new
  locations and delete old paths. **`./pa` is the one canonical entry**; drop
  `python main.py` as an advertised path.
- **Staged**, not one big-bang commit.

## Open — confirm in the fresh session's plan mode (recommendations noted)
- **Package layout (recommended):** one top-level package `portfolioanalyzer/`,
  **keeping filenames** to limit churn, with subpackages. *Not* `src/` (extra layer
  the `./pa` workflow doesn't need). Proposed grouping:
  - `portfolioanalyzer/`: `main.py`, `metrics.py`, `utils.py`, `data_loader.py`,
    `data_update_cli.py`
  - `portfolioanalyzer/loaders/`, `portfolioanalyzer/timeseries/` (moved in)
  - `portfolioanalyzer/assets/`: `sgb_holdings`, `sgb_tranches`, `fund_lifecycle`,
    `synthetic_civ`, `civ_to_returns`, `bond_calculators`
  - `portfolioanalyzer/calc/`: `portfolio_calculator`
  - `portfolioanalyzer/reporting/`: `visualizer`, `drawdowns_csv`, `output_metadata`
- **Orphaned data (recommended: move to `attic/orphaned-data/` with a README; not
  delete).** The 13 zero-reference CSVs (verified by repo-wide grep):
  `INDIR3TIB01STM.csv`, `NIFTRI.csv`, `rbi_91day_tbills.csv`, and the 10
  `*_Historical_PR_*.csv` index dumps.
- **`data/` split (recommended):** `data/reference/` (NIFTY TRI, INDIRLTLT01STM,
  gold_monthly_inr, legacy risk-free) + `data/funds/` (vro_funds, fund_catalog,
  ppf_interest_rates, sgb_tranches).
- **CLI-combo configs (recommended): move the 16 `output_csv-*` / `…show_plot…`
  TOMLs to `tests/fixtures/cli_configs/`** (they're CLI hand-test fixtures, not user
  config). Keep `config/example_config.toml` + `config/mid-cap_config.toml`.
- **Loose top-level files:** `no_benchmark_config.toml` (1 line) → fold into
  `config/` or a fixture; `requirements.txt` → keep (documented convenience mirror)
  or drop; `output_metadata.py` → into the package (above).
- **`port/`:** keep the name (prior decision) unless you say otherwise.

## Code/paths that MUST be rewritten with the moves (the blast radius)
- Every intra-module import (`import metrics`, `from loaders… import`, `from
  timeseries… import`, `from main import …`) and every **test** import.
- `pyproject.toml`: `packages = ["loaders"]` → the new package; `[project.scripts]`
  `main:cli` / `data_update_cli:main` → `portfolioanalyzer.main:cli` etc.
- `pa` wrapper: exec the new entry directly (no `main.py` shim).
- Every `data/...csv` path in code, `config/*.toml`, `port/*.toml`, and the goldens
  (`tests/golden/replay/reference/…`, `tests/fixtures/golden_master_config.toml`).
- Docs trees: README "Project Structure", ARCHITECTURE module map, QUICKSTART.
- Path globs in `tests/unit/test_docs_consistency.py` (config glob) and
  `tests/unit/test_fund_catalog.py` (catalog path).

## Also fix during the reorg (surfaced 2026-06-20)
- **Doc-guard gap:** `tests/unit/test_docs_consistency.py` validates flags/keys but
  not the *invocation form* — a bare `portfolio-analyzer …` (needs an activated
  venv) passed review. Extend it to flag bare console-script invocations in the
  docs and require `./pa …` (or `./venv/bin/…`). (README examples themselves were
  already corrected to `./pa` in a small pre-reorg fix.)

## After the reorg (separate, gated)
- **Wire the 10 fetchable funds into VRO parity** (`data/funds/fund_catalog.csv` →
  `data/funds/vro_funds.csv`, needs VRO plan ids) so `test_vro_parity.py` validates their
  Beta/Alpha.
- **Build new `port/*.toml` test portfolios** from catalogued funds.

## Verification each stage
Full unit suite + the 1e-9 goldens (the real safety net for path/move breakage),
ruff, mypy, and the doc-consistency + fund-catalog guards; then the network-gated
tier before merge.
