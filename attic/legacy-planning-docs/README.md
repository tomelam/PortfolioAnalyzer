# Legacy planning docs (archived 2026-06-19)

These eight Markdown files are **pre-refactor, ChatGPT-era planning notes** that
describe a superseded design (the `TimeseriesFrame` / `metrics_calculator` layout,
before the `loaders/` and `timeseries/` packages and the pure-function `metrics.py`).

They were moved out of `docs/` during **Thread 4 (docs staleness sweep)** of the
2026-06 cleanup round: zero references from any canonical doc or from code, and
their content no longer matches the codebase. Kept here for historical record
rather than deleted.

| File | Was |
|---|---|
| `STRUCTURE.md` | early module-structure sketch |
| `TODO.md` | early global TODO list |
| `Variables.md` | variable-naming notes |
| `Calculations.md` | calculation notes |
| `Calculation_styles.md` | daily-vs-monthly calculation-style notes |
| `Class_Roles_Summary.md` | proposed class responsibilities |
| `RefactorTable.md` | refactor mapping table |
| `full_global_todos.md` | aggregated TODOs |

Current canonical docs live in `docs/` (ARCHITECTURE, TESTING, CONTRIBUTING,
DATA_REFRESH, OUTPUTS) and the repo root (README, QUICKSTART). The live task
board is `KANBAN.md`.
