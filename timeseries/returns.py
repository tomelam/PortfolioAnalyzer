import warnings

import numpy as np
import pandas as pd

from utils import dbg


class TimeseriesReturn:
    """
    Lightweight wrapper around a Pandas Series for strict, validated time series analysis.

    Provides methods for performance metrics like CAGR, Sharpe, Sortino, alpha, beta, and max drawdowns,
    assuming the internal series represents daily returns or cumulative NAVs.

    Only accepts a pandas Series as input. Fails fast on incorrect types.

    Note: a number of unused reporting/alignment helpers were parked in
    ``attic/timeseries_return_helpers.py`` on 2026-06-17 (see KANBAN Phase E
    port candidates); this class now exposes only its live surface.
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
        import metrics
        return metrics.cagr(self.value_series())

    def volatility(
            self,
            periods_per_year: int = 252,
            frequency: str = "daily"
    ) -> float:
        """Annualized volatility. Delegates to ``metrics.volatility``."""
        import metrics
        returns, scale = self._standardized_returns(frequency, periods_per_year)
        return metrics.volatility(returns, periods_per_year=scale)

    def sortino(
            self,
            risk_free_rate: float = 0.0,
            periods_per_year: int = 252,
            frequency: str = "daily"
    ) -> float:
        """Annualized Sortino ratio. Delegates to ``metrics.sortino``."""
        import metrics
        returns, scale = self._standardized_returns(frequency, periods_per_year)
        return metrics.sortino(returns, risk_free_rate=risk_free_rate, periods_per_year=scale)

    def sharpe(
            self,
            risk_free_rate: float = 0.0,
            periods_per_year: int = 252,
            frequency: str = "daily"
    ) -> float:
        """Annualized Sharpe ratio. Delegates to ``metrics.sharpe``."""
        import metrics
        returns, scale = self._standardized_returns(frequency, periods_per_year)
        return metrics.sharpe(returns, risk_free_rate=risk_free_rate, periods_per_year=scale)

    def max_drawdown(self):
        """Maximum drawdown of the price series. Delegates to ``metrics.max_drawdown``."""
        import metrics
        return metrics.max_drawdown(self.value_series())

    def max_drawdowns(self, threshold=0.05):
        """List the fully-recovered drawdowns whose magnitude exceeds ``threshold``.

        Delegates to ``metrics.max_drawdowns``. Returns a list of dicts
        with ``start_date``, ``trough_date``, ``recovery_date``, and
        ``drawdown`` (positive fraction).
        """
        import metrics
        return metrics.max_drawdowns(self.value_series(), threshold=threshold)

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
                beta = self.beta_regression(benchmark_ret)
            else:
                raise

        mean_rp = y_aligned.mean()
        mean_rb = x_aligned.mean()
        alpha_daily = mean_rp - (risk_free_rate + beta * (mean_rb - risk_free_rate))
        alpha = (1 + alpha_daily) ** 252 - 1
        return alpha

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
