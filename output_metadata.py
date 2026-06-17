"""Format run metadata for embedding in / alongside the PNG snapshot.

Pure string formatters — no matplotlib, no I/O — so they're trivially unit
tested. ``main.py`` uses these to:

- print the reference-data provenance to stderr (for logs / posterity),
- render a small provenance footnote onto the plot (the *visible* copy), and
- embed a labeled metrics block + provenance in the PNG's ``tEXt`` chunks (the
  *invisible*, machine-recoverable copy — far more usable than the positional
  metrics CSV row).

PNG ``tEXt`` values are Latin-1; :func:`build_png_metadata` coerces every value
so an exotic character in a fund name can't make ``savefig`` raise.
"""

from __future__ import annotations

NA = "n/a"


def _short_ts(value: str | None) -> str:
    """A timestamp/date string trimmed to its ``YYYY-MM-DD`` date, or ``n/a``."""
    if not value:
        return NA
    return str(value)[:10]


def format_kv_block(pairs: list[tuple[str, str]]) -> str:
    """Render ``(label, value)`` pairs as a colon-aligned block."""
    width = max((len(k) for k, _ in pairs), default=0)
    return "\n".join(f"{k + ':':<{width + 1}} {v}" for k, v in pairs)


def format_provenance(provenance: list[dict]) -> str:
    """Full per-source provenance: last data point, when fetched, last attempt."""
    lines = []
    for e in provenance:
        lines.append(
            f"{e['label']}: "
            f"last {e.get('last_date') or NA} | "
            f"fetched {e.get('fetched_at') or NA} | "
            f"attempted {e.get('attempted_at') or NA}"
        )
    return "\n".join(lines)


def format_provenance_footnote(provenance: list[dict], *, run_label: str) -> str:
    """Compact one-block provenance for the visible plot footnote (dates only)."""
    parts = [
        f"{e['label']}: last {_short_ts(e.get('last_date'))}, "
        f"fetched {_short_ts(e.get('fetched_at'))}"
        for e in provenance
    ]
    return f"Reference data ({run_label}) — " + "   ·   ".join(parts)


def _latin1_safe(value) -> str:
    """Coerce to a Latin-1-encodable string (PNG tEXt requirement)."""
    return str(value).encode("latin-1", "replace").decode("latin-1")


def build_png_metadata(
    *,
    portfolio_label: str,
    run_label: str,
    generated: str,
    metric_pairs: list[tuple[str, str]],
    provenance: list[dict],
) -> dict[str, str]:
    """Assemble the PNG ``tEXt`` dict: context + a human-readable metrics block
    + reference-data provenance. Every value is made Latin-1-safe."""
    md = {
        "portfolio": portfolio_label,
        "run": run_label,
        "generated": generated,
        "metrics": format_kv_block(metric_pairs),
        "reference_data": format_provenance(provenance),
    }
    return {k: _latin1_safe(v) for k, v in md.items()}
