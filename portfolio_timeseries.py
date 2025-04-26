from typing import Dict, Optional
import pandas as pd
from asset_timeseries import AssetTimeseries, from_civ
from utils import dbg

class PortfolioTimeseries:
    """
    Represents a collection of AssetTimeseries objects,
    combined into a unified portfolio timeseries.
    """
    def __init__(self, assets: Dict[str, AssetTimeseries], weights: Optional[Dict[str, float]] = None):
        if not assets:
            raise ValueError("PortfolioTimeseries requires at least one asset")
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
            ret = asset.ret.value_series().dropna() * weight
            # Fail-fast if any missing in the raw return series
            ret_series = asset.ret.value_series()
            weight = self.weights.get(name, 0.0)
            ret = ret_series.dropna() * weight
            weighted_returns.append(ret)

        if not weighted_returns:
            return pd.Series(dtype=float)

        df = pd.concat(weighted_returns, axis=1, join="inner")
        # Guard against silently computing with missing values
        if df.isnull().values.any():
            raise ValueError("Combined portfolio returns contain NaNs")
        return df.sum(axis=1).sort_index()

    def combined_civ_series(self) -> pd.Series:
        """
        True portfolio CIV: weighted sum of each asset's original CIV series.
        """
        weighted_navs = []
        for name, asset in self.assets.items():
            w = self.weights.get(name, 0.0)
            nav = asset.civ.value_series() * w
            weighted_navs.append(nav)

        if not weighted_navs:
            return pd.Series(dtype=float)

        df = pd.concat(weighted_navs, axis=1, join="inner")
        return df.sum(axis=1).sort_index()


def from_multiple_nav_series(
    nav_dict: Dict[str, Optional[pd.Series]],
    weights: Optional[Dict[str, float]] = None
) -> PortfolioTimeseries:
    """
    Convert a dict of raw NAV series into a PortfolioTimeseries instance.

    Parameters:
        nav_dict (dict): {"fund_name": pd.Series or None}
        weights (dict): Optional weights for each asset

    Returns:
        PortfolioTimeseries: The constructed portfolio
    """
    assets: Dict[str, AssetTimeseries] = {}
    for name, series in nav_dict.items():
        if series is None:
            dbg(f"Skipping '{name}': no data")
            continue
        dbg(f"Checking '{name}': {type(series)}")
        if not isinstance(series, pd.Series):
            dbg(f"⚠️ Skipping '{name}': not a Series (got {type(series)})")
            continue
        assets[name] = from_civ(series)
    dbg("Returning from `from_multiple_nav_series`")
    return PortfolioTimeseries(assets=assets, weights=weights)
