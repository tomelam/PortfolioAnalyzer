"""Parked code retained for provenance.

Modules in this package are intentionally **not imported** by the live
package, **not linted** (see ``[tool.ruff] extend-exclude`` in
pyproject.toml), and **not counted** in coverage. They preserve functions
that once had a rationale but are currently unused, so they can be revived
deliberately (see KANBAN.md → Phase E port items) rather than recovered
from git archaeology.

Nothing here is guaranteed to run as-is; some helpers reference an older
TimeseriesReturn surface (``self.columns`` / ``self.shape`` /
``self.interpolate`` / ``self.annualized``) that the current class does
not provide. Reviving a helper means re-wiring it to today's API.
"""
