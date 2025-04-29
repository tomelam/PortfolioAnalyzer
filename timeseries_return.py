import pandas as pd
import numpy as np

class TimeseriesReturn:
    """
    Represents a standardized return series.
    Handles metrics like Sharpe, Sortino, volatility, CAGR, drawdowns, etc.
    Input: pd.Series of returns (daily or monthly).
    """

    def __init__(self, series: pd.Series):
        if not isinstance(series, pd.Series):
            raise TypeError(f"Expected a pandas Series, got {type(series)}")
        if series.name != "value":
            raise ValueError(f"Expected series name 'value', got {series.name}")
        if not np.issubdtype(series.dtype, np.number):
            raise ValueError("TimeseriesReturn requires numeric data.")
        self.series = series.dropna()

    def volatility(self, frequency: str = "monthly") -> float:
        returns = self.series
        if frequency == "monthly":
            scale = 12
        else:
            scale = 252  # daily
        return returns.std() * np.sqrt(scale)

    def sharpe(self, risk_free_rate=0.0, frequency="daily", periods_per_year=252) -> float:
        """
         Calculate the Sharpe ratio based on periodic excess returns.
         Scales result to annualized Sharpe ratio.
         """
        if frequency == "monthly":
            returns = self.series.resample("ME").last().pct_change().dropna()
            scale = 12
        else:
            returns = self.series.pct_change().dropna()
            scale = periods_per_year
        excess_returns = returns - (risk_free_rate / scale)
        std_dev = returns.std()
        if std_dev < 1e-12:
            if excess_returns.mean() > 0:
                return float("inf")
            elif excess_returns.mean() < 0:
                return float("-inf")
            else:
                return 0.0
        return (excess_returns.mean() * scale) / (std_dev * scale**0.5)

    def sortino(self, risk_free_rate=0.0, frequency="daily", periods_per_year=252) -> float:
        """
        Calculate the Sortino ratio using downside deviation.
        Annualized according to frequency.
        """
        if frequency == "monthly":
            returns = self.series.resample("ME").last().pct_change().dropna()
            scale = 12
        else:
            returns = self.series.pct_change().dropna()
            scale = periods_per_year
        excess_returns = returns - (risk_free_rate / scale)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        if downside_std < 1e-12:
            if excess_returns.mean() > 0:
                return float("inf")
            elif excess_returns.mean() < 0:
                return float("-inf")
            else:
                return 0.0
        return (excess_returns.mean() * scale) / (downside_std * scale**0.5)

    def cagr(self) -> float:
        # This assumes monthly returns, consistent with Morningstar method
        cumulative_return = (1 + self.series).prod() - 1
        periods = len(self.series)
        if periods == 0:
            return float("nan")
        years = periods / 12  # monthly returns assumed
        return (1 + cumulative_return) ** (1 / years) - 1

    def alpha_capm(self, benchmark: "TimeseriesReturn", risk_free_rate: float = 0.0) -> float:
        """
        CAPM alpha: actual excess return - expected return from beta * benchmark.
        Both series must be aligned and in returns.
        """
        x = benchmark.series.align(self.series, join="inner")[0].dropna()
        y = self.series.align(benchmark.series, join="inner")[0].dropna()

        if len(x) < 3:
            raise ValueError("Too few data points to compute alpha.")

        excess_x = x - risk_free_rate
        excess_y = y - risk_free_rate

        beta = excess_y.cov(excess_x) / excess_x.var()
        expected_return = risk_free_rate + beta * excess_x.mean()
        actual_return = y.mean()
        return actual_return - expected_return

    def alpha_capm(self, benchmark: "TimeseriesReturn", risk_free_rate: float = 0.0) -> float:
        """
        CAPM alpha: actual excess return - expected return from beta * benchmark.
        Both series must be aligned and in returns.
        """
        x = benchmark.series.align(self.series, join="inner")[0].dropna()
        y = self.series.align(benchmark.series, join="inner")[0].dropna()

        if len(x) < 3:
            raise ValueError("Too few data points to compute alpha.")

        excess_x = x - risk_free_rate
        excess_y = y - risk_free_rate

        beta = excess_y.cov(excess_x) / excess_x.var()
        expected_return = risk_free_rate + beta * excess_x.mean()
        actual_return = y.mean()
        return actual_return - expected_return

    def beta_capm(self, benchmark: "TimeseriesReturn", risk_free_rate: float = 0.0) -> float:
        """
        Calculate beta (CAPM-style) relative to the benchmark.
        """
        excess_self = self.series - risk_free_rate
        excess_bench = benchmark.series - risk_free_rate
        aligned_self, aligned_bench = excess_self.align(excess_bench, join="inner")

        if aligned_self.std() < 1e-12:
            return float("nan")  # undefined when no variance

        return aligned_self.cov(aligned_bench) / aligned_bench.var()

