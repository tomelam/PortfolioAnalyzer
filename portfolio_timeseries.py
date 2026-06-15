from typing import Dict, Optional
import pandas as pd
from asset_timeseries import AssetTimeseries, from_civ
from timeseries_civ import TimeseriesCIV
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
        if total_weight != 1:
            raise ValueError("PortfolioTimeseries asset weights must sum to 1")
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

    def combined_civ_series(self) -> TimeseriesCIV:
        """
        True portfolio CIV: weighted sum of each asset's NORMALIZED CIV series.

        Each asset's CIV is rebased to 1.0 at the common start date before
        applying its weight. Without this, an asset with a raw NAV of 2500
        (e.g. PPF) would dominate one with NAV 18 (an MF) regardless of
        intended allocation.
        """
        if not self.assets:
            return TimeseriesCIV(pd.Series(dtype=float))

        # First inner-join all raw CIVs so we can normalize from a common start.
        raw = pd.concat(
            {name: asset.civ.value_series() for name, asset in self.assets.items()},
            axis=1,
            join="inner",
        ).sort_index()

        if raw.empty:
            return TimeseriesCIV(pd.Series(dtype=float))

        normalized = raw.divide(raw.iloc[0])
        weights = pd.Series(self.weights).reindex(normalized.columns).fillna(0.0)
        combined = normalized.mul(weights, axis=1).sum(axis=1)
        combined.name = "value"
        return TimeseriesCIV(combined)


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


def civ_and_returns(portfolio_ts: PortfolioTimeseries) -> tuple[pd.Series, pd.Series]:
    """Conveniently get both the CIV and daily return series."""
    return portfolio_ts.combined_civ_series(), portfolio_ts.combined_daily_returns()

