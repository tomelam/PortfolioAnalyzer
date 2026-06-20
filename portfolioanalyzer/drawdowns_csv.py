"""Write a per-drawdown CSV alongside the single-row portfolio metrics CSV.

The single-row metrics CSV reports only the *worst* drawdown of a
portfolio. Users who want the full picture get every recovered (and the
final unrecovered) drawdown via the sibling ``<portfolio>.drawdowns.csv``.

Record schema is whatever ``TimeseriesCIV.max_drawdowns`` emits:

- ``start``, ``trough``, ``end`` — dates (Timestamp or date)
- ``drawdown`` — positive fraction (e.g. ``0.1923`` for 19.23%)
- ``drawdown_days`` — int
- ``recovery_days`` — int, or ``None`` for the final unrecovered drawdown
"""

from __future__ import annotations

import csv
from collections.abc import Iterable

_HEADER = [
    "start_date",
    "trough_date",
    "recovery_date",
    "depth_pct",
    "drawdown_days",
    "recovery_days",
]


def _fmt_date(d) -> str:
    """ISO date string, or '' for None."""
    if d is None:
        return ""
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)


def write_drawdowns_csv(drawdowns: Iterable[dict], path: str) -> None:
    """Write a CSV with one row per drawdown to ``path``.

    Columns: ``start_date, trough_date, recovery_date, depth_pct,
    drawdown_days, recovery_days``. ``depth_pct`` is the human-friendly
    negative-percent form (``-19.23`` for a 19.23% drawdown). Unrecovered
    drawdowns get empty strings for ``recovery_date`` and ``recovery_days``.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_HEADER)
        for d in drawdowns:
            w.writerow([
                _fmt_date(d.get("start")),
                _fmt_date(d.get("trough")),
                _fmt_date(d.get("end") if d.get("recovery_days") is not None else None),
                f"{-d['drawdown'] * 100:.2f}",
                d.get("drawdown_days", ""),
                "" if d.get("recovery_days") is None else d["recovery_days"],
            ])
