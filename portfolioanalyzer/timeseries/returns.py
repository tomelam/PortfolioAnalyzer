import numpy as np
import pandas as pd

from portfolioanalyzer import metrics
from portfolioanalyzer.utils import info


def _summary_metrics(ts, risk_free_rate=0.0, frequency="daily"):
    """Ordered ``(label, value, is_percent)`` metric rows for reporting helpers.

    The current API's annualized-return notion is CAGR (geometric) and its
    annualized-volatility notion is ``volatility()`` — these replace the parked
    code's removed ``annualized()`` dict. ``is_percent`` drives display
    formatting: CAGR / Max Drawdown / Volatility are fractions shown as
    percents; Sharpe / Sortino are dimensionless ratios shown as-is.
    """
    return [
        ("CAGR", ts.cagr(), True),
        ("Max Drawdown", ts.max_drawdown(), True),
        ("Volatility", ts.volatility(frequency=frequency), True),
        ("Sharpe", ts.sharpe(risk_free_rate, frequency=frequency), False),
        ("Sortino", ts.sortino(risk_free_rate, frequency=frequency), False),
    ]


class TimeseriesReturn:
    """
    Lightweight wrapper around a Pandas Series for strict, validated time series analysis.

    Provides methods for performance metrics like CAGR, Sharpe, Sortino, alpha, beta, and max drawdowns,
    assuming the internal series represents daily returns or cumulative NAVs.

    Only accepts a pandas Series as input. Fails fast on incorrect types.

    Note: the reporting helpers below (``info_summary`` / ``describe_as_report``
    / ``to_csv_report`` / ``to_latex_table`` / ``compare_to`` / ``as_rolling``)
    were revived from ``attic/timeseries_return_helpers.py`` in 2026-06. The
    remaining parked alignment helpers (``align_with`` / ``clip_to_overlap`` /
    ``aligned_to`` / ``interpolated``) stay parked until cross-asset analysis
    needs them.
    """
    @property
    def name(self) -> str:
        """
        Return the name of the underlying series.
        """
        return self._series.name

    @property
    def index(self) -> pd.Index:
        """
        Return the index of the underlying series.
        """
        return self._series.index

    @property
    def values(self) -> np.ndarray:
        """
        Return the raw numpy array of the underlying series values.
        """
        return self._series.values

    @property
    def iloc(self):
        """
        Integer-location based indexing (like Series.iloc).
        """
        return self._series.iloc

    @property
    def loc(self):
        """
        Label-based indexing (like Series.loc).
        """
        return self._series.loc

    def _standardized_returns(self, frequency: str, periods_per_year: int) -> tuple[pd.Series, int]:
        """
        Get returns and scaling factor depending on frequency and testing mode.
        """
        if frequency == "monthly":
            returns = self._series.resample("ME").last().pct_change().dropna()
            scale = 12 if periods_per_year == 252 else periods_per_year
        else:  # daily
            returns = self._series.pct_change().dropna()
            scale = periods_per_year

        return returns, scale

    def set_series(self, new_series: pd.Series):
        if not isinstance(new_series, pd.Series):
            raise TypeError("Timeseries expects a pd.Series")
        self._series = new_series.sort_index()

    def dropna(self) -> "TimeseriesReturn":
        """
        Return a new TimeseriesReturn with missing values dropped.
        """
        return TimeseriesReturn(self._series.dropna())

    def mean(self) -> float:
        """
        Return the mean of the underlying series.
        """
        return self._series.mean()

    def __init__(self, series: pd.Series):
        if not isinstance(series, pd.Series):
            raise TypeError(f"TimeseriesReturn expects a pd.Series, got {type(series)}")
        if series.name != "value":
            series = series.rename("value")
        self._series = series.sort_index()

    def value_series(self):
        if self._series.name != "value":
            raise ValueError(f"Expected series name to be 'value', got {self._series.name}")
        return self._series

    def cagr(self):
        """Annualized return of the underlying value series.

        Delegates to ``metrics.cagr`` (the underlying ``_series`` is
        treated as a price/CIV series).
        """
        return metrics.cagr(self.value_series())

    def volatility(
            self,
            periods_per_year: int = 252,
            frequency: str = "daily"
    ) -> float:
        """Annualized volatility. Delegates to ``metrics.volatility``."""
        returns, scale = self._standardized_returns(frequency, periods_per_year)
        return metrics.volatility(returns, periods_per_year=scale)

    def sortino(
            self,
            risk_free_rate: float = 0.0,
            periods_per_year: int = 252,
            frequency: str = "daily"
    ) -> float:
        """Annualized Sortino ratio. Delegates to ``metrics.sortino``."""
        returns, scale = self._standardized_returns(frequency, periods_per_year)
        return metrics.sortino(returns, risk_free_rate=risk_free_rate, periods_per_year=scale)

    def sharpe(
            self,
            risk_free_rate: float = 0.0,
            periods_per_year: int = 252,
            frequency: str = "daily"
    ) -> float:
        """Annualized Sharpe ratio. Delegates to ``metrics.sharpe``."""
        returns, scale = self._standardized_returns(frequency, periods_per_year)
        return metrics.sharpe(returns, risk_free_rate=risk_free_rate, periods_per_year=scale)

    def max_drawdown(self):
        """Maximum drawdown of the price series. Delegates to ``metrics.max_drawdown``."""
        return metrics.max_drawdown(self.value_series())

    def max_drawdowns(self, threshold=0.05):
        """List the fully-recovered drawdowns whose magnitude exceeds ``threshold``.

        Delegates to ``metrics.max_drawdowns``. Returns a list of dicts
        with ``start_date``, ``trough_date``, ``recovery_date``, and
        ``drawdown`` (positive fraction).
        """
        return metrics.max_drawdowns(self.value_series(), threshold=threshold)

    def alpha_regression(self, benchmark_ret: "TimeseriesReturn") -> float:
        """Regression alpha (the intercept). Delegates to ``metrics.alpha_regression``.

        Both series are assumed to be daily returns; they are aligned by date.
        """
        return metrics.alpha_regression(self.value_series(), benchmark_ret.value_series())

    def alpha_capm(
            self,
            benchmark_ret: "TimeseriesReturn",
            risk_free_rate: float = 0.0,
            fallback_to_simple_beta: bool = False
    ) -> float:
        """Annualized Jensen's alpha (CAPM). Delegates to ``metrics.alpha_capm``."""
        return metrics.alpha_capm(
            self.value_series(),
            benchmark_ret.value_series(),
            risk_free_rate=risk_free_rate,
            fallback_to_simple_beta=fallback_to_simple_beta,
        )

    def beta_regression(self, benchmark_ret: "TimeseriesReturn") -> float:
        """Regression beta (the slope). Delegates to ``metrics.beta_regression``."""
        return metrics.beta_regression(self.value_series(), benchmark_ret.value_series())

    def beta_capm(self, benchmark_ret: "TimeseriesReturn", risk_free_rate: float = 0.0) -> float:
        """CAPM beta. Delegates to ``metrics.beta_capm``."""
        return metrics.beta_capm(
            self.value_series(), benchmark_ret.value_series(), risk_free_rate=risk_free_rate
        )

    # --- reporting helpers (revived from attic/, 2026-06) ------------------

    def info_summary(self, name: str = "Timeseries") -> None:
        """Print a compact structural summary (shape, range, NaNs) to stderr."""
        s = self.value_series()
        info(f"{name} shape: {s.shape}")
        info(f"{name} date range: {self.index.min().date()} → {self.index.max().date()}")
        info(f"NaNs in 'value': {int(s.isna().sum())}")
        info(f"Non-zero values: {int((s != 0).sum())}")

    def describe_as_report(self, name: str = "Timeseries") -> None:
        """Print descriptive statistics of the value series to stderr."""
        s = self.value_series()
        info(f"🧾 Report: {name}")
        info(f"- Date range: {self.index.min().date()} → {self.index.max().date()}")
        info(f"- Observations: {len(s)}")
        info(f"- Missing: {int(s.isna().sum())}")
        info(f"- Non-zero: {int((s != 0).sum())}")
        info(f"- Mean: {s.mean():.6f}")
        info(f"- Std Dev: {s.std():.6f}")
        info(f"- Min: {s.min():.6f}")
        info(f"- Max: {s.max():.6f}")

    def to_csv_report(self, path, name: str = "Timeseries") -> None:
        """Write a one-row summary CSV (structure + descriptive stats)."""
        s = self.value_series()
        summary = {
            "name": name,
            "start_date": self.index.min().date(),
            "end_date": self.index.max().date(),
            "observations": len(s),
            "missing": int(s.isna().sum()),
            "nonzero": int((s != 0).sum()),
            "mean": s.mean(),
            "std_dev": s.std(),
            "min": s.min(),
            "max": s.max(),
        }
        pd.DataFrame([summary]).to_csv(path, index=False)

    def to_latex_table(
        self,
        compare_to: "TimeseriesReturn | None" = None,
        name: str = "Series",
        title: str | None = None,
        label: str | None = None,
    ) -> str:
        """Return LaTeX for a metrics table (CAGR / Max DD / Vol / Sharpe / Sortino).

        If ``compare_to`` is another ``TimeseriesReturn``, emit a 2-column
        comparison. Percent metrics render with ``\\%``; ratios render plain.
        """
        from io import StringIO

        rows_self = _summary_metrics(self)
        rows_other = _summary_metrics(compare_to) if compare_to is not None else None

        def fmt(value: float, is_percent: bool) -> str:
            return f"{value * 100:.2f}\\%" if is_percent else f"{value:.2f}"

        buffer = StringIO()
        buffer.write("\\begin{table}[ht]\n\\centering\n")
        if title:
            buffer.write(f"\\caption{{{title}}}\n")
        if label:
            buffer.write(f"\\label{{{label}}}\n")

        columns = "lrr" if rows_other else "lr"
        buffer.write(f"\\begin{{tabular}}{{{columns}}}\n\\toprule\n")
        buffer.write("Metric & " + name)
        if rows_other:
            buffer.write(" & Comparison")
        buffer.write(" \\\\\n\\midrule\n")

        for i, (metric, val1, is_percent) in enumerate(rows_self):
            if rows_other:
                val2 = rows_other[i][1]
                buffer.write(f"{metric} & {fmt(val1, is_percent)} & {fmt(val2, is_percent)} \\\\\n")
            else:
                buffer.write(f"{metric} & {fmt(val1, is_percent)} \\\\\n")

        buffer.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        return buffer.getvalue()

    def compare_to(
        self,
        other: "TimeseriesReturn",
        name_self: str = "This",
        name_other: str = "Other",
        risk_free_rate: float = 0.0,
        frequency: str = "daily",
    ) -> None:
        """Print a side-by-side metrics comparison (over the common dates) to stderr.

        Raises ``TypeError`` if ``other`` is not a ``TimeseriesReturn``, and
        ``ValueError`` if either series is empty or the two share fewer than 30
        dates (too little overlap to compare meaningfully).
        """
        if not isinstance(other, TimeseriesReturn):
            raise TypeError("Expected TimeseriesReturn")
        s1 = self.value_series().dropna()
        s2 = other.value_series().dropna()
        if s1.empty or s2.empty:
            raise ValueError("Cannot compare empty series.")
        common = s1.index.intersection(s2.index)
        if len(common) < 30:
            raise ValueError("Too little overlap between series to compare meaningfully.")

        rows1 = _summary_metrics(TimeseriesReturn(s1.loc[common]), risk_free_rate, frequency)
        rows2 = _summary_metrics(TimeseriesReturn(s2.loc[common]), risk_free_rate, frequency)

        info(f"\n📊 Comparison: {name_self} vs {name_other}\n")
        info(f"{'Metric':<20} | {name_self:<12} | {name_other}")
        info("-" * 50)
        for (metric, v1, is_percent), (_, v2, _) in zip(rows1, rows2, strict=True):
            f1 = f"{v1 * 100:.2f}%" if is_percent else f"{v1:.2f}"
            f2 = f"{v2 * 100:.2f}%" if is_percent else f"{v2:.2f}"
            info(f"{metric:<20} | {f1:<12} | {f2}")

    def as_rolling(self, window: int = 30, method: str = "mean") -> "TimeseriesReturn":
        """Return a new ``TimeseriesReturn`` of a rolling metric over 'value'.

        Supported ``method``: 'mean', 'std', 'median', 'min', 'max'.
        """
        if method not in {"mean", "std", "median", "min", "max"}:
            raise ValueError(f"Unsupported method: {method}")
        func = getattr(self.value_series().rolling(window), method)
        return TimeseriesReturn(func())
