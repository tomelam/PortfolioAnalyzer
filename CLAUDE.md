# PortfolioAnalyzer — Working Rules for Claude

Two files load above this one and are not repeated here: `~/.claude/CLAUDE.md` (universal rules — fail loud, investigate before delete, branch/merge discipline,
in-repo wrappers, temp files, stall protocol) and `~/Projects/meta/pipeline-conventions.md`.
**Only this project's own rules and its local instances of the shared ones belong here.**

## Entry point

Run via the in-repo wrapper: `./pa <args>` (executes `venv/bin/python main.py`). The package is `portfolioanalyzer/`; it can also be invoked as `python -m portfolioanalyzer.main`. See `docs/QUICKSTART.md` and `docs/ARCHITECTURE.md`.

---

Staged-pipeline conventions shared with the other `make`-driven projects here — make vs
staleness, `.PRECIOUS` vs `.DELETE_ON_ERROR:`, one-argument stages, atomic resume,
sanity-checking a produced artifact: `~/Projects/meta/pipeline-conventions.md`.

---

## Code and data rules

### Fail loud — this project's instance
*(the general rule is in `~/.claude/CLAUDE.md`)*
If a designed automated fetch/refresh fails, let it stay stale and let the freshness gate block. Under `--allow-stale`, degraded artifacts must explicitly say they are degraded AND why (the failure, not just the symptom). The project is block-by-default.

### Free-market native-unit pricing
Price each asset using the most open, free, and fair market, in that market's **native currency** — and do **not** FX-convert it. Every output is normalized growth or a ratio (CAGR, vol, Sharpe, Alpha/Beta), never an absolute INR figure. A constant rescale (troy-oz→gram) cancels under normalization; a time-varying FX path (USD→INR) does **not** cancel — it injects the rupee's managed depreciation into what should be the asset's own return.

**Exception — contractual cash values:** when the number *is* a defined payout (e.g. SGB premature-redemption price is legally the IBJA 999-gold INR figure), use the actual contractual source. That is measuring realized cash, not performance.

### No yfinance
Do not propose `yfinance` (or similar unofficial Yahoo Finance scrapers) as a dependency. It wraps an undocumented endpoint that breaks without warning. Prefer a documented public API (FRED, LBMA, an exchange's own feed), a static CSV the user refreshes manually, or a paid feed. If yfinance is the only path to a feature, surface that as a trade-off and ask before committing.

### Bumping webgrab

`webgrab` is the shared fetching library behind every loader here, installed from GitHub
and **pinned to a tag** — never a bare branch URL. An unpinned `git+` dependency tracks
HEAD, so two installs on different days get different fetching code while every other
runtime dependency here is pinned exactly and there is no lock file. It moved five times
in one evening on 2026-09-07.

To bump it:

1. In `~/Projects/webgrab`: get its own suite green (`make test`), then tag —
   `git tag -a vX.Y.Z -m "…"` — and `git push origin vX.Y.Z`.
2. **Check the tag resolves over anonymous HTTPS**, which is what pip uses:
   `git ls-remote https://github.com/tomelam/webgrab.git vX.Y.Z`. Pushing over the SSH
   remote and reading it back over SSH proves nothing about a fresh installer.
3. Update the pin in `pyproject.toml`, reinstall, and run **the full suite including the
   network tier** — the fetchers are the whole point of the dependency, and only the
   network tier exercises them.
4. Commit the pin bump on its own, saying which webgrab commits it takes.

Three tests hold this, all verified by breaking them (2026-09-08):
`test_non_pypi_dependencies_are_pinned_to_a_ref` fails any `git+` dependency with no
`@ref`; `test_readme_names_every_runtime_dependency_not_exempted` fails if the README's
install section stops naming a declared dependency; and
`tests/integration/test_webgrab_pin_current.py` (network tier) fails when webgrab
publishes a **newer tag** — the nudge to run this ritual. It ignores HEAD movement, which
is constant, and clears itself once the pin is bumped.

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

### The merge gate, concretely
*(branch/merge discipline itself is in `~/.claude/CLAUDE.md`)*
The full suite here means the network tier too: `pytest -m "not network or network"`, which includes the live niftyindices and VRO browser tests. All green is the gate.

### Commit messages
End every commit message with the Co-Authored-By trailer naming **whichever model made the
change** — the harness supplies the real line. Never hard-code a version: this said
`Claude Sonnet 4.6` until 2026-09-08 while the last 20 commits had all used Opus 5.
```
Co-Authored-By: Claude <model> <noreply@anthropic.com>
```

---

## Process preferences

### Tidy before big jobs
When sequencing a backlog, put small clean-and-tidy refactors and closeable items first; defer large, research-heavy, or externally-dependent work to the end. Tidy foundations make the big jobs cleaner.

### Autonomous mode
When the user explicitly authorizes a long batch ("continue without my intervention", "proceed until all KANBAN tasks are done"), apply these defaults for the session:

- **No confirmation prompts** on individual items. Pick the next item, do it, commit, move on.
- **Minimize shell commands.** Pytest runs are essential; cosmetic checks (`make help`, `--help`) are not.
- **Destructive/shared actions still require explicit naming** — don't push to remote, delete branches, or change CI unless the user explicitly names those actions in the authorization.
- **Commit per logical cycle**, not at the very end.
- Reset to "ask before risky actions" at the start of each new session.
