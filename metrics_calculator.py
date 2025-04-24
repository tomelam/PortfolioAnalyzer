from logging import debug
import numpy as np
import pandas as pd
from utils import dbg
from timeseries import TimeseriesFrame


def alpha_from_series(ret: TimeseriesFrame, benchmark_ret: TimeseriesFrame) -> float:
    """
    Calculate Jensen's alpha: excess return unexplained by benchmark.
    Assumes daily returns. Aligned by date.
    """
    x = benchmark_ret.value_series()
    y = ret.value_series()
    x, y = x.align(y, join="inner")
    if len(x) < 2:
        raise ValueError("Alpha requires at least two aligned data points")
    coeffs = np.polyfit(x, y, 1)
    beta = coeffs[0]
    alpha = coeffs[1]
    return float(alpha)


def beta_from_series(ret: TimeseriesFrame, benchmark_ret: TimeseriesFrame) -> float:
    """
    Calculate beta: sensitivity of returns to benchmark returns.
    Assumes daily returns. Aligned by date.
    """
    x = benchmark_ret.value_series()
    y = ret.value_series()
    x, y = x.align(y, join="inner")
    if len(x) < 2:
        raise ValueError("Beta requires at least two aligned data points")
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])
