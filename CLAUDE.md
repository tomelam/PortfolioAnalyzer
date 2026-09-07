# PortfolioAnalyzer — Working Rules for Claude

## Entry point

Run via the in-repo wrapper: `./pa <args>` (executes `venv/bin/python main.py`). The package is `portfolioanalyzer/`; it can also be invoked as `python -m portfolioanalyzer.main`. See `docs/QUICKSTART.md` and `docs/ARCHITECTURE.md`.

---

## Code and data rules

### Fail loud — no silent fallbacks
If a designed automated fetch/refresh fails, let it stay stale and let the freshness gate block. Do **not** add a manual workaround that quietly routes around the failure, and do **not** present a degraded artifact with neutral framing. Under `--allow-stale`, degraded artifacts must explicitly say they are degraded AND why (the failure, not just the symptom). The project is block-by-default; a silent fallback defeats the whole point.

### Free-market native-unit pricing
Price each asset using the most open, free, and fair market, in that market's **native currency** — and do **not** FX-convert it. Every output is normalized growth or a ratio (CAGR, vol, Sharpe, Alpha/Beta), never an absolute INR figure. A constant rescale (troy-oz→gram) cancels under normalization; a time-varying FX path (USD→INR) does **not** cancel — it injects the rupee's managed depreciation into what should be the asset's own return.

**Exception — contractual cash values:** when the number *is* a defined payout (e.g. SGB premature-redemption price is legally the IBJA 999-gold INR figure), use the actual contractual source. That is measuring realized cash, not performance.

### No yfinance
Do not propose `yfinance` (or similar unofficial Yahoo Finance scrapers) as a dependency. It wraps an undocumented endpoint that breaks without warning. Prefer a documented public API (FRED, LBMA, an exchange's own feed), a static CSV the user refreshes manually, or a paid feed. If yfinance is the only path to a feature, surface that as a trade-off and ask before committing.

### Self-contained tooling
Do not install or depend on anything outside the project directory: no cron/launchd jobs, no `~/.local/bin` symlinks, no mandatory `source venv/bin/activate`. Prefer the `./pa` wrapper. If a scheduler ever seems needed, raise it as a question — don't set it up.

### Self-documenting outputs
Output artifacts should carry their own run context/provenance — embed it durably in the file (PNG `tEXt` metadata, header comment, clearly-labeled block), not only in transient stdout. Freshness/provenance chatter belongs on **stderr** so stdout stays clean for the report/CSV. Keep additions additive; don't break existing schemas.

---

## Discovery and diagnostics

### Keep throwaway diagnostics
When a one-off script establishes a non-obvious finding about an external site or data source (a "discovery spike"), **keep it and document it**. Park it under `scripts/` with a docstring stating the conclusion; list it in `scripts/README.md` under a clearly labelled "discovery spikes" section (not run by CI); distil the conclusion into `docs/ARCHITECTURE.md`. Don't delete it once it has answered its question.

### Docs accuracy
Treat README/QUICKSTART prose as **unverified until checked against the code**. Before asserting a CLI flag, config key, or data file path, verify it against `main.build_parser()`, the `config.get("…")` calls in `main.build_settings`, or the actual loaders. The automated guard is `tests/unit/test_docs_consistency.py` — **extend it** whenever you add a CLI flag, config key, or documented data file.

---

## Git workflow

### One branch per logical thread
Work on a short-lived branch off `main` — one branch per logical thread. Commit cleanly and focused within the branch.

### Full suite gates every merge
Before merging, run the full test suite **including the network tier**: `pytest -m "not network or network"`. This includes live niftyindices + VRO browser tests. All green is the gate.

### Ask before each merge
After tests pass, **pause and ask** before merging and pushing. The user treats the merge as an explicit checkpoint. Merge with `--no-ff` (explicit merge commit), then delete the branch.

### Commit messages
End every commit message with the Co-Authored-By trailer:
```
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

---

## Process preferences

### Investigate before delete
Before proposing to delete or overwrite **anything** you did not create (tags, branches, files, data): **investigate it first** (`git show`, read the file, check references) and report what it is. Default to **keep**. Only raise removal with concrete evidence it is safe and worthless, and let the user decide. "Looks like junk from its name" is not investigation.

### Tidy before big jobs
When sequencing a backlog, put small clean-and-tidy refactors and closeable items first; defer large, research-heavy, or externally-dependent work to the end. Tidy foundations make the big jobs cleaner.

### Autonomous mode
When the user explicitly authorizes a long batch ("continue without my intervention", "proceed until all KANBAN tasks are done"), apply these defaults for the session:

- **No confirmation prompts** on individual items. Pick the next item, do it, commit, move on.
- **Minimize shell commands.** Pytest runs are essential; cosmetic checks (`make help`, `--help`) are not.
- **Destructive/shared actions still require explicit naming** — don't push to remote, delete branches, or change CI unless the user explicitly names those actions in the authorization.
- **Commit per logical cycle**, not at the very end.
- Reset to "ask before risky actions" at the start of each new session.
