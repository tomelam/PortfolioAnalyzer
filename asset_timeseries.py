from dataclasses import dataclass
import pandas as pd
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

def from_nav(nav_series: pd.Series) -> AssetTimeseries:
    """
    Convert a NAV or unit-value Series into an AssetTimeseries.
    Assumes a datetime index in ascending order.

    Parameters:
        nav_series (pd.Series): investment value series

    Returns:
        AssetTimeseries
    """
    nav = nav_series.sort_index().dropna()
    ret = nav.pct_change().fillna(0)
    cumret = (1 + ret).cumprod()

    return AssetTimeseries(
        civ=TimeseriesFrame(nav.to_frame("value")),
        ret=TimeseriesFrame(ret.to_frame("value")),
        cumret=TimeseriesFrame(cumret.to_frame("value")),
    )
