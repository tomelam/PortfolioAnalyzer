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


class PortfolioTimeseries:
    """
    Represents a collection of AssetTimeseries objects,
    combined into a unified portfolio timeseries.
    """
    def __init__(self, assets: Dict[str, AssetTimeseries], weights: Optional[Dict[str, float]] = None):
        self.assets = assets
        self.weights = weights or {name: 1.0 for name in assets}

        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def combined_daily_returns(self) -> pd.Series:
        """
        Weighted sum of daily returns across all assets.
        """
        weighted_returns = []
        for name, asset in self.assets.items():
            weight = self.weights.get(name, 0.0)
            ret = asset.ret.value_series()
            if ret.isnull().any():
                raise ValueError(f"Return series for asset '{name}' contains NaNs")
            weighted_returns.append(ret * weight)

        if not weighted_returns:
            return pd.Series(dtype=float)

        return pd.concat(weighted_returns, axis=1).sum(axis=1).sort_index()

    def combined_nav_series(self, base: float = 1.0) -> pd.Series:
        """
        Reconstruct cumulative NAV from combined returns.
        """
        returns = self.combined_daily_returns()
        return (1 + returns).cumprod() * base


def from_multiple_nav_series(nav_dict: Dict[str, Optional[pd.Series]], weights: Optional[Dict[str, float]] = None) -> PortfolioTimeseries:
    """
    Convert a dict of raw NAV series into a PortfolioTimeseries instance.

    Parameters:
        nav_dict (dict): {"fund_name": pd.Series or None}
        weights (dict): Optional weights for each asset

    Returns:
        PortfolioTimeseries: The constructed portfolio
    """
    assets = {}
    for name, series in nav_dict.items():
        if series is not None:
            assets[name] = from_civ(series)
    return PortfolioTimeseries(assets=assets, weights=weights)
