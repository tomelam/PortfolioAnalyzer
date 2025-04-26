from typing import Dict, Optional
import pandas as pd
from dataclasses import dataclass
from timeseries import TimeseriesFrame

@dataclass
class AssetTimeseries:
    """
    Unified representation of an asset's value and return series.

    - civ: Current Investment Value series (NAV/unit value)
    - ret: Daily return series (simple pct-change)
    - cumret: Cumulative returns (compounded growth)
    """
    civ: TimeseriesFrame
    ret: TimeseriesFrame
    cumret: TimeseriesFrame

    def summary(self):
        return {
            "Start": self.civ.index.min().strftime("%Y-%m-%d"),
            "End": self.civ.index.max().strftime("%Y-%m-%d"),
            "Start Value": self.civ.value_series().iloc[0],
            "End Value": self.civ.value_series().iloc[-1],
            "Total Return": self.cumret.value_series().iloc[-1],
        }


def from_civ(nav_series: pd.Series) -> AssetTimeseries:
    """
    Build AssetTimeseries from a raw NAV-like Series.
    """
    if nav_series is None:
        raise ValueError("NAV series is None")
    if not isinstance(nav_series, pd.Series):
        raise TypeError("Expected a pandas Series for NAV data")
    if nav_series.isnull().any():
        raise ValueError("NAV series contains NaNs")
    if not isinstance(nav_series.index, pd.DatetimeIndex):
        raise TypeError("NAV series must have a DatetimeIndex")

    nav_series = nav_series.sort_index()
    nav_series.name = "value"

    civ = TimeseriesFrame(nav_series)
    ret = TimeseriesFrame(nav_series.pct_change().dropna().rename("value"))
    cumret = TimeseriesFrame(((1 + ret.value_series()).cumprod()).rename("value"))

    return AssetTimeseries(civ=civ, ret=ret, cumret=cumret)
