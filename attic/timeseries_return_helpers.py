"""Parked TimeseriesReturn helpers (extracted 2026-06-17).

These methods lived on ``timeseries.returns.TimeseriesReturn`` but had zero
production callers and were dragging the live class's test coverage down.
They are preserved here verbatim because they once had a clear rationale
and are named as future-port candidates in KANBAN.md → Phase E:

- alignment helpers: ``align_with``, ``clip_to_overlap``, ``aligned_to``,
  ``interpolated``
- misc series transforms: ``rolling_mean``, ``percent_change``, ``plot_with``

The reporting helpers (``info_summary``, ``describe_as_report``,
``to_csv_report``, ``to_latex_table``, ``compare_to``, ``as_rolling``) were
revived onto the live ``TimeseriesReturn`` in 2026-06 (Thread 6) and removed
from here; see ``timeseries/returns.py`` + ``tests/unit/test_reporting_helpers.py``.

Status: UNWIRED. Several reference an older surface the current
TimeseriesReturn no longer exposes — ``self.columns``, ``self.shape``,
``self['value']`` (no ``__getitem__``), ``self.interpolate``,
``self.annualized`` — so they would raise ``AttributeError`` if called
today. Reviving one means re-wiring it to the current API and adding tests.

They are kept as methods on a parked class purely so the ``self``
references remain legible; this class is never instantiated.
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (used by parked helpers)
import pandas as pd

from utils import dbg, info  # noqa: F401  (used by parked helpers)


class ParkedTimeseriesReturnHelpers:
    """Never imported by live code. See module docstring."""

    # --- alignment helpers (KANBAN Phase E port candidates) ---

    def align_with(self, other, how="inner"):
        """Align with another series or DataFrame."""
        aligned_self, aligned_other = self.align(other, join=how)
        from timeseries.returns import TimeseriesReturn
        return TimeseriesReturn(aligned_self), aligned_other

    def clip_to_overlap(self, other):
        """Clip to the overlapping date range. NaNs allowed post-alignment."""
        overlap = self.index.intersection(other.index)
        if overlap.empty:
            raise ValueError("No overlapping dates between series.")
        return self.loc[overlap], other.loc[overlap]

    def aligned_to(self, reference):
        """Interpolate to match index of a reference TimeseriesReturn."""
        from timeseries.returns import TimeseriesReturn
        df = self.interpolate(method="linear", limit_direction="both")
        return TimeseriesReturn(
            df.reindex(reference.index).interpolate(method="linear", limit_direction="both")
        )

    def interpolated(self, method="time"):
        """Return a new TimeseriesReturn with missing values filled by interpolation."""
        from timeseries.returns import TimeseriesReturn
        df = self.interpolate(method=method, limit_direction="both")
        return TimeseriesReturn(df)

    # --- misc series transforms ---

    def rolling_mean(self, window):
        """Return a new TimeseriesReturn with 'value' replaced by its rolling mean."""
        from timeseries.returns import TimeseriesReturn
        if "value" not in self.columns:
            raise KeyError("No 'value' column found.")
        s = self.value_series().rolling(window=window, min_periods=1).mean()
        return TimeseriesReturn(pd.DataFrame({"value": s}, index=self.index))

    def percent_change(self):
        """Return a new TimeseriesReturn with percent change of 'value'."""
        from timeseries.returns import TimeseriesReturn
        s = self.value_series().pct_change()
        return TimeseriesReturn(pd.DataFrame({"value": s}, index=self.index))

    def plot_with(self, other_series, title=None):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        self.value_series().plot(ax=ax, label="This timeseries", style="-")
        other_series.plot(ax=ax, label="Other", style="--")

        ax.set_title(title or "Timeseries Comparison")
        ax.set_ylabel("Value")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()


# --- Reference: original commented-out max_drawdowns implementation ---
# Retained from timeseries/returns.py. The live max_drawdowns now delegates
# to metrics.max_drawdowns; this loop-based version is kept only for
# provenance / cross-checking the recovered logic.
#
#     Calculate maximum drawdowns with full retracements. A drawdown is
#     recorded only after the series has fully recovered to the peak from
#     which it fell.
#
#     cumulative = self.value_series()
#     gain_peak = cumulative.cummax()
#     max_drawdowns = []
#     in_drawdown = False
#     drawdown_start_date = None
#     drawdown_start_value = None
#     trough_date = None
#     trough_value = None
#     for date, value in cumulative.items():
#         current_peak = gain_peak.at[date]
#         if value < current_peak:
#             if not in_drawdown:
#                 in_drawdown = True
#                 drawdown_start_date = date
#                 drawdown_start_value = current_peak
#                 trough_date = date
#                 trough_value = value
#             else:
#                 if value < trough_value:
#                     trough_date = date
#                     trough_value = value
#         else:
#             if in_drawdown:
#                 if value >= drawdown_start_value:
#                     drawdown_percentage = (trough_value - drawdown_start_value) / drawdown_start_value
#                     if abs(drawdown_percentage) >= threshold:
#                         max_drawdowns.append({
#                             "start_date": drawdown_start_date,
#                             "trough_date": trough_date,
#                             "recovery_date": date,
#                             "depth_pct": drawdown_percentage,
#                             "trough_value": trough_value,
#                             "recovery_value": self.value_series().loc[date],
#                             "drawdown": drawdown_percentage * 100,
#                             "drawdown_days": (trough_date - drawdown_start_date).days + 1,
#                             "recovery_days": (date - drawdown_start_date).days + 1,
#                         })
#                     in_drawdown = False
#     return max_drawdowns
