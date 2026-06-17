"""Parked TimeseriesReturn helpers (extracted 2026-06-17).

These methods lived on ``timeseries.returns.TimeseriesReturn`` but had zero
production callers and were dragging the live class's test coverage down.
They are preserved here verbatim because they once had a clear rationale
and are named as future-port candidates in KANBAN.md → Phase E:

- reporting helpers: ``info_summary``, ``describe_as_report``,
  ``to_csv_report``, ``to_latex_table``, ``compare_to``, ``as_rolling``
- alignment helpers: ``align_with``, ``clip_to_overlap``, ``aligned_to``,
  ``interpolated``
- misc series transforms: ``rolling_mean``, ``percent_change``, ``plot_with``

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

    def as_rolling(self, window=30, method="mean"):
        """Return a new TimeseriesReturn with a rolling metric applied to 'value'.

        Supported methods: 'mean', 'std', 'median', 'min', 'max'.
        """
        from timeseries.returns import TimeseriesReturn
        if method not in {"mean", "std", "median", "min", "max"}:
            raise ValueError(f"Unsupported method: {method}")
        func = getattr(self.value_series().rolling(window), method)
        return TimeseriesReturn(pd.DataFrame({"value": func()}, index=self.index))

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

    # --- reporting helpers (KANBAN Phase E port candidates) ---

    def info_summary(self, name="Timeseries"):
        from utils import info
        info(f"{name} shape: {self.shape}")
        info(f"{name} date range: {self.index.min().date()} → {self.index.max().date()}")
        info(f"NaNs in 'value': {self['value'].isna().sum()}")
        info(f"Non-zero values: {(self['value'] != 0).sum()}")

    def describe_as_report(self, name="Timeseries"):
        from utils import info

        s = self.value_series()
        info(f"🧾 Report: {name}")
        info(f"- Date range: {self.index.min().date()} → {self.index.max().date()}")
        info(f"- Observations: {len(s)}")
        info(f"- Missing: {s.isna().sum()}")
        info(f"- Non-zero: {(s != 0).sum()}")
        info(f"- Mean: {s.mean():.6f}")
        info(f"- Std Dev: {s.std():.6f}")
        info(f"- Min: {s.min():.6f}")
        info(f"- Max: {s.max():.6f}")

    def to_csv_report(self, path, name="Timeseries"):
        s = self.value_series()
        summary = {
            "name": name,
            "start_date": self.index.min().date(),
            "end_date": self.index.max().date(),
            "observations": len(s),
            "missing": s.isna().sum(),
            "nonzero": (s != 0).sum(),
            "mean": s.mean(),
            "std_dev": s.std(),
            "min": s.min(),
            "max": s.max(),
        }
        pd.DataFrame([summary]).to_csv(path, index=False)

    def to_latex_table(self, compare_to=None, name="Series", title=None, label=None):
        """Return LaTeX code for a table of metrics for this series.

        If compare_to is another TimeseriesReturn, generate a 2-column comparison.
        """
        from io import StringIO

        def summary(ts):
            return {
                "CAGR": ts.cagr(),
                "Max Drawdown": ts.max_drawdown(),
                "Ann Return": ts.annualized()["annualized_return"],
                "Ann Volatility": ts.annualized()["annualized_volatility"],
                "Sharpe": ts.sharpe(),
                "Sortino": ts.sortino(),
            }

        metrics_self = summary(self)
        metrics_other = summary(compare_to) if compare_to else None

        buffer = StringIO()
        buffer.write("\\begin{table}[ht]\n\\centering\n")
        if title:
            buffer.write(f"\\caption{{{title}}}\n")
        if label:
            buffer.write(f"\\label{{{label}}}\n")

        columns = f"{'lrr' if metrics_other else 'lr'}"
        buffer.write(f"\\begin{{tabular}}{{{columns}}}\n\\toprule\n")
        buffer.write("Metric & " + name)
        if compare_to:
            buffer.write(" & Comparison")
        buffer.write(" \\\\\n\\midrule\n")

        for metric in metrics_self:
            val1 = metrics_self[metric]
            val1_fmt = (
                f"{val1*100:.2f}\\%"
                if "return" in metric.lower() or "drawdown" in metric.lower()
                else f"{val1:.2f}"
            )
            if metrics_other:
                val2 = metrics_other[metric]
                val2_fmt = (
                    f"{val2*100:.2f}\\%"
                    if "return" in metric.lower() or "drawdown" in metric.lower()
                    else f"{val2:.2f}"
                )
                buffer.write(f"{metric} & {val1_fmt} & {val2_fmt} \\\\\n")
            else:
                buffer.write(f"{metric} & {val1_fmt} \\\\\n")

        buffer.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        return buffer.getvalue()

    def compare_to(
        self, other, name_self="This", name_other="Other", risk_free_rate=0.0, frequency="daily"
    ):
        from timeseries.returns import TimeseriesReturn
        assert isinstance(other, TimeseriesReturn), "Expected TimeseriesReturn"
        if self.value_series().empty or other.value_series().empty:
            raise ValueError("Cannot compare empty series.")
        if "value" not in self.columns or "value" not in other.columns:
            raise KeyError("Missing 'value' column in one of the series.")
        s1 = self.value_series().dropna()
        s2 = other.value_series().dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 30:
            raise ValueError("Too little overlap between series to compare meaningfully.")
        s1 = s1.loc[common]
        s2 = s2.loc[common]
        ts1 = TimeseriesReturn(pd.DataFrame({"value": s1}, index=common))
        ts2 = TimeseriesReturn(pd.DataFrame({"value": s2}, index=common))

        def summary(ts):
            return {
                "CAGR": ts.cagr(),
                "Max Drawdown": ts.max_drawdown(),
                "Ann Return": ts.annualized(frequency)["annualized_return"],
                "Ann Vol": ts.annualized(frequency)["annualized_volatility"],
                "Sharpe": ts.sharpe(risk_free_rate, frequency),
                "Sortino": ts.sortino(risk_free_rate, frequency),
            }

        p1 = summary(ts1)
        p2 = summary(ts2)
        info(f"\n📊 Comparison: {name_self} vs {name_other}\n")
        info(f"{'Metric':<20} | {name_self:<12} | {name_other}")
        info("-" * 50)
        for k in p1:
            v1 = f"{p1[k]*100:.2f}%" if "drawdown" in k.lower() or "return" in k.lower() else f"{p1[k]:.2f}"
            v2 = f"{p2[k]*100:.2f}%" if "drawdown" in k.lower() or "return" in k.lower() else f"{p2[k]:.2f}"
            info(f"{k:<20} | {v1:<12} | {v2}")


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
