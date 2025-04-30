from matplotlib.lines import drawStyles
import pandas as pd
import numpy as np
import warnings
from dateutil.relativedelta import relativedelta
from utils import info  # if you use `info()` for logging


class TimeseriesReturn:
    """
    Lightweight wrapper around a Pandas Series for strict, validated time series analysis.

    Provides methods for performance metrics like CAGR, Sharpe, Sortino, alpha, beta, and max drawdowns,
    assuming the internal series represents daily returns or cumulative NAVs.
    
    Only accepts a pandas Series as input. Fails fast on incorrect types.
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
        returns = self._series.dropna()

        if frequency == "monthly":
            returns = self._series.resample("ME").last().pct_change().dropna()
            scale = 12 if periods_per_year == 252 else periods_per_year
        else:  # daily
            scale = periods_per_year
        #print("🔵 NAV series head:\n", self._series.head(10))
        #print("🔵 Returns head:\n", returns.head(10))
        #print("🔵 Scaling factor:", scale)

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

    def align_with(self, other: pd.Series | pd.DataFrame, how="inner"):
        """Align with another series or DataFrame."""
        aligned_self, aligned_other = self.align(other, join=how)
        return TimeseriesReturn(aligned_self), aligned_other

    def clip_to_overlap(self, other: pd.Series | pd.DataFrame):
        """
        Clip to the overlapping date range.
        NaNs are allowed post-alignment.
        """
        overlap = self.index.intersection(other.index)
        if overlap.empty:
            raise TimeseriesValidationError("No overlapping dates between series.")
        return self.loc[overlap], other.loc[overlap]

    def value_series(self):
        if self._series.name != "value":
            raise ValueError(f"Expected series name to be 'value', got {self._series.name}")
        return self._series

    def info_summary(self, name="Timeseries"):
        from utils import info
        info(f"{name} shape: {self.shape}")
        info(f"{name} date range: {self.index.min().date()} → {self.index.max().date()}")
        info(f"NaNs in 'value': {self['value'].isna().sum()}")
        info(f"Non-zero values: {(self['value'] != 0).sum()}")
    
    def aligned_to(self, reference):
        """
        Interpolate to match index of a reference TimeseriesReturn.
        """
        df = self.interpolate(method="linear", limit_direction="both")
        return TimeseriesReturn(df.reindex(reference.index).interpolate(method="linear", limit_direction="both"))

    def interpolated(self, method="time"):
        """
        Return a new TimeseriesReturn with missing values (floats only) filled using interpolation.
        Does not modify original.
        """
        df = self.interpolate(method=method, limit_direction="both")
        return TimeseriesReturn(df)

    # Optional
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

    # Optional
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

    # Optional
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
            "max": s.max()
        }
        pd.DataFrame([summary]).to_csv(path, index=False)

    # Optional
    def rolling_mean(self, window):
        """
        Return a new TimeseriesReturn with 'value' replaced by its rolling mean.
        """
        if "value" not in self.columns:
            raise KeyError("No 'value' column found.")
        s = self.value_series().rolling(window=window, min_periods=1).mean()
        return TimeseriesReturn(pd.DataFrame({"value": s}, index=self.index))

    def percent_change(self):
        """
        Return a new TimeseriesReturn with percent change of 'value'.
        """
        s = self.value_series().pct_change()
        return TimeseriesReturn(pd.DataFrame({"value": s}, index=self.index))

    def cagr(self):
        """
        Compute the compound annual growth rate (CAGR) from the first to last 'value'.
        Assumes a price-like cumulative series.
        This method uses relativedelta (actual calendar time). It “knows” that months
        vary in length (and handles end‑of‑month rollovers and leap days) when you use
        its months or years parameters.
        """
        s = self.value_series().dropna()
        if len(s) < 2:
            raise ValueError("CAGR calculation requires at least two data points.")

        start_value = s.iloc[0]
        end_value = s.iloc[-1]
        start_date = s.index[0]
        end_date = s.index[-1]

        days = (end_date - start_date).days
        years = days / 365.25
        
        if years <= 0:
            raise ValueError("CAGR calculation requires a positive time span.")
        if start_value == 0:
            raise ValueError("CAGR calculation invalid: start value is zero.")

        return (end_value / start_value) ** (1 / years) - 1

    def volatility(
            self,
            periods_per_year: int = 252,
            frequency: str = "daily"
    ) -> float:
        """
        Calculate the annualized volatility (standard deviation of returns).

        If frequency='monthly', automatically resamples returns to monthly, adjusts scaling,
        and ignores periods_per_year.
        For deterministic unit testing, use periods_per_year=1 to disable annualization.
        """
        returns, scale = self._standardized_returns(frequency, periods_per_year)

        print("DEBUG: returns head", returns.head())
        print("DEBUG: returns std dev", returns.std())
        print("DEBUG: scale", scale)
        print("DEBUG: final volatility", returns.std() * (scale ** 0.5))
        return returns.std() * (scale ** 0.5)

    def sortino(
            self,
            risk_free_rate: float = 0.0,
            periods_per_year: int = 252,
            frequency: str = "daily"
    ) -> float:
        """
        Calculate the annualized Sortino ratio.

        If frequency='monthly', automatically resamples returns to monthly, adjusts scaling,
        and ignores periods_per_year.
        Set periods_per_year=1 to disable annualization for exact unit-testable ratios.
        """
        returns, scale = self._standardized_returns(frequency, periods_per_year)

        excess_returns = returns - (risk_free_rate / scale)
        fuzz = -1e-10
        downside_returns = excess_returns[excess_returns < fuzz]

        if downside_returns.std() == 0:
            if excess_returns.mean() > 0:
                return float("inf")
            elif excess_returns.mean() < 0:
                return float("-inf")
            else:
                return 0.0

        return (excess_returns.mean() * scale) / (downside_returns.std() * (scale ** 0.5))

    def sharpe(
            self,
            risk_free_rate: float = 0.0,
            periods_per_year: int = 252,
            frequency: str = "daily"
    ) -> float:
        """
        Calculate the annualized Sharpe ratio.

        If frequency='monthly', automatically resamples returns to monthly, adjusts scaling,
        and ignores periods_per_year.
        Set periods_per_year=1 to disable annualization for exact unit-testable ratios.
        """
        returns, scale = self._standardized_returns(frequency, periods_per_year)

        excess_returns = returns - (risk_free_rate / scale)
        std_dev = returns.std()

        if std_dev < 1e-12:  # treat as zero to avoid absurdly large ratios
            if excess_returns.mean() > 0:
                return float("inf")   # perfectly stable outperformance
            elif excess_returns.mean() < 0:
                return float("-inf")  # perfectly stable underperformance
            else:
                return 0.0            # no return = no reward

        return (excess_returns.mean() * scale) / (std_dev * (scale ** 0.5))

    def max_drawdown(self):
        """
        Compute the maximum drawdown of a cumulative series.
        Assumes 'value' is NAV or price.
        """
        s = self.value_series().dropna()
        if s.empty:
            raise ValueError("Max drawdown requires at least one data point.")
        cumulative_max = s.cummax()
        drawdowns = (s - cumulative_max) / cumulative_max
        return drawdowns.min()  # most negative value

    """
    def max_drawdowns(self, threshold=0.05):
        
        Calculate maximum drawdowns with full retracements.
        A drawdown is recorded only after the series has fully recovered to the peak from which it fell.
        
        Parameters:
            cumulative (TimeseriesReturn): Cumulative portfolio returns.
            threshold (float): Minimum drawdown percentage to report (e.g., 0.05 for 5%).

        Returns:
            list of dict: Each dict contains 'start_date', 'trough_date', 'recovery_date',
                and 'drawdown' (percentage).
        
        # Ensure we work with a Series.
        cumulative = self.value_series()

        gain_peak = cumulative.cummax()
        max_drawdowns = []
        in_drawdown = False
        drawdown_start_date = None
        drawdown_start_value = None
        trough_date = None
        trough_value = None

        for date, value in cumulative.items():
            current_peak = gain_peak.at[date]
            if value < current_peak:
                # We're in a drawdown.
                if not in_drawdown:
                    in_drawdown = True
                    drawdown_start_date = date
                    drawdown_start_value = current_peak  # Record the peak value at drawdown start
                    trough_date = date
                    trough_value = value
                else:
                    # Update trough if value falls further.
                    if value < trough_value:
                        trough_date = date
                        trough_value = value
            else:
                # Recovery: series has reached (or exceeded) the previous peak.
                if in_drawdown:
                    if value >= drawdown_start_value:  # Full retracement to the initial peak
                        drawdown_percentage = (trough_value - drawdown_start_value) / drawdown_start_value
                        if abs(drawdown_percentage) >= threshold:
                            max_drawdowns.append({
                                "start_date": drawdown_start_date,
                                "trough_date": trough_date,
                                "recovery_date": date,
                                "depth_pct": drawdown_percentage,
                                "trough_value": trough_value,
                                "recovery_value": self.value_series().loc[date],
                                "drawdown": drawdown_percentage * 100,
                                "drawdown_days": (trough_date - drawdown_start_date).days + 1,
                                "recovery_days": (date - drawdown_start_date).days + 1,
                            })
                        in_drawdown = False
        # Do not record drawdowns that haven't been recovered.
        return max_drawdowns
    """

    def alpha_regression(self, benchmark_ret: "TimeseriesReturn") -> float:
        """
        Calculate regression alpha (not Jensen's alpha).

        This uses a linear regression of the portfolio's daily returns against the benchmark's:
            y = alpha + beta * x

        The intercept (alpha) represents the portfolio's average daily return
        when the benchmark's return is zero.

        Assumes both series are daily returns and aligned by date.
        """
        x = benchmark_ret.value_series()
        y = self.value_series()
        if x.empty or y.empty:
            raise ValueError("Cannot compute alpha: input series is empty.")

        if np.allclose(x, x.iloc[0]) or np.allclose(y, y.iloc[0]):
            raise ValueError("Cannot compute alpha: no variation in returns.")

        # Align series and warn if any mismatch occurred
        x_aligned, y_aligned = x.align(y, join="inner")
        num_trimmed = max(len(x) - len(x_aligned), len(y) - len(y_aligned))
        if num_trimmed > 0:
            dbg(f"alpha(): alignment trimmed {num_trimmed} entries from return series.")

        if len(x_aligned) < 3:
            raise ValueError("Too few aligned data points after trimming. Cannot compute alpha.")

        # Linear regression: y = alpha + beta * x
        with warnings.catch_warnings():
            warnings.simplefilter("error", np.RankWarning)
            coeffs = np.polyfit(x_aligned, y_aligned, 1)

        return float(coeffs[1])  # Intercept = alpha

    def alpha_capm(
            self,
            benchmark_ret: "TimeseriesReturn",
            risk_free_rate: float = 0.0,
            fallback_to_simple_beta: bool = False
    ) -> float:
        """
        Calculate Jensen's alpha using CAPM:

            alpha = mean(Rp) - [Rf + beta * (mean(Rb) - Rf)]

        Where:
        - Rp = portfolio daily returns (self)
        - Rb = benchmark daily returns (benchmark_ret)
        - Rf = scalar risk-free daily return
        - beta = covariance(Rp, Rb) / variance(Rb)

        Assumes both series are aligned on dates and expressed as daily returns.
        Falls back to simple beta() if beta_capm() fails and fallback is enabled.
        """
        x = benchmark_ret.value_series()
        y = self.value_series()

        if x.empty or y.empty:
            raise ValueError("Cannot compute alpha_capm: input series is empty.")

        x_aligned, y_aligned = x.align(y, join="inner")
        if len(x_aligned) < 3:
            raise ValueError("Too few aligned data points to compute alpha_capm.")

        try:
            beta = self.beta_capm(benchmark_ret, risk_free_rate=risk_free_rate)
        except Exception as e:
            if fallback_to_simple_beta:
                dbg(f"⚠️ beta_capm() failed, falling back to beta(): {e}")
                beta = self.beta(benchmark_ret)
            else:
                raise

        excess_x = x_aligned - risk_free_rate
        expected_return = risk_free_rate + beta * excess_x.mean()
        actual_return = y_aligned.mean()
        return actual_return - expected_return

    def beta_regression(self, benchmark_ret: "TimeseriesReturn") -> float:
        """
        Calculate beta: sensitivity of returns to benchmark returns.
        Assumes daily returns. Aligned by date.
        """
        x = benchmark_ret.value_series()
        y = self.value_series()
        x, y = x.align(y, join="inner")
        if len(x) < 2:
            raise ValueError("Beta requires at least two aligned data points")
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0])

    def beta_capm(self, benchmark_ret: "TimeseriesReturn", risk_free_rate: float = 0.0) -> float:
        """
        Calculate CAPM-style beta:

        beta = Cov(Rp - Rf, Rb - Rf) / Var(Rb - Rf)

        Where:
        - Rp = portfolio daily returns (self)
        - Rb = benchmark daily returns (benchmark_ret)
        - Rf = scalar risk-free daily return

        Assumes both series are aligned on dates and expressed as daily returns.
        """
        x = benchmark_ret.value_series()
        y = self.value_series()

        if x.empty or y.empty:
            raise ValueError("Cannot compute beta_capm: input series is empty.")

        x_aligned, y_aligned = x.align(y, join="inner")
        if len(x_aligned) < 3:
            raise ValueError("Too few aligned data points to compute beta_capm.")

        excess_x = x_aligned - risk_free_rate
        excess_y = y_aligned - risk_free_rate

        return excess_y.cov(excess_x) / excess_x.var()

    def as_rolling(self, window=30, method="mean"):
        """
        Return a new TimeseriesReturn with rolling metric applied to 'value'.
        Supported methods: 'mean', 'std', 'median', 'min', 'max'
        """
        if method not in {"mean", "std", "median", "min", "max"}:
            raise ValueError(f"Unsupported method: {method}")
        func = getattr(self.value_series().rolling(window), method)
        return TimeseriesReturn(pd.DataFrame({"value": func()}, index=self.index))

    def to_latex_table(self, compare_to=None, name="Series", title=None, label=None):
        """
        Return LaTeX code for a table of metrics for this series.
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
            buffer.write(f" & Comparison")
        buffer.write(" \\\\\n\\midrule\n")

        for metric in metrics_self:
            val1 = metrics_self[metric]
            val1_fmt = f"{val1*100:.2f}\\%" if 'return' in metric.lower() or 'drawdown' in metric.lower() else f"{val1:.2f}"
            if metrics_other:
                val2 = metrics_other[metric]
                val2_fmt = f"{val2*100:.2f}\\%" if 'return' in metric.lower() or 'drawdown' in metric.lower() else f"{val2:.2f}"
                buffer.write(f"{metric} & {val1_fmt} & {val2_fmt} \\\\\n")
            else:
                buffer.write(f"{metric} & {val1_fmt} \\\\\n")

        buffer.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        return buffer.getvalue()

    def compare_to(self, other, name_self="This", name_other="Other", risk_free_rate=0.0, frequency="daily"):
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
            v1 = f"{p1[k]*100:.2f}%" if 'drawdown' in k.lower() or 'return' in k.lower() else f"{p1[k]:.2f}"
            v2 = f"{p2[k]*100:.2f}%" if 'drawdown' in k.lower() or 'return' in k.lower() else f"{p2[k]:.2f}"
            info(f"{k:<20} | {v1:<12} | {v2}")

