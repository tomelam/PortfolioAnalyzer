import numpy as np
import pandas as pd

import metrics


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
