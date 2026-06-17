
import pandas as pd

from asset_timeseries import AssetTimeseries, from_civ
from timeseries_civ import TimeseriesCIV
from utils import dbg


class PortfolioTimeseries:
    """
    Represents a collection of AssetTimeseries objects,
    combined into a unified portfolio timeseries.
    """
    def __init__(self, assets: dict[str, AssetTimeseries], weights: dict[str, float] | None = None):
        if not assets:
            raise ValueError("PortfolioTimeseries requires at least one asset")
        self.assets = assets
        self.weights = weights or {name: 1.0 for name in assets}

        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        # Tolerance matches data_loader.validate_allocations (1% band).
        # Hand-typed TOML fractions routinely sum to 0.9999999999999999
        # (IEEE-754 rounding) or 1.0000000000000002; the strict `!= 1`
        # check rejected legitimate portfolios spuriously.
        if abs(total_weight - 1) > 0.01:
            raise ValueError(
                f"PortfolioTimeseries asset weights must sum to 1 "
                f"(got {total_weight:.6f})"
            )
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def effective_window(self) -> dict:
        """Return the portfolio's effective time window and the assets that set it.

        Bounded by the latest asset inception (``start``) and the earliest
        asset last-NAV (``end``). The accompanying ``*_limited_by`` keys
        name which asset sat at each bound — useful for the stdout banner
        that explains why a portfolio with a defunct fund plots up to
        that fund's last reported NAV rather than today.
        """
        if not self.assets:
            raise ValueError("PortfolioTimeseries has no assets")
        starts = {n: a.civ.value_series().index.min() for n, a in self.assets.items()}
        ends = {n: a.civ.value_series().index.max() for n, a in self.assets.items()}
        start_name, start_date = max(starts.items(), key=lambda kv: kv[1])
        end_name, end_date = min(ends.items(), key=lambda kv: kv[1])
        return {
            "start": start_date,
            "end": end_date,
            "start_limited_by": start_name,
            "end_limited_by": end_name,
        }

    def combined_civ_series(self) -> TimeseriesCIV:
        """
        True portfolio CIV: weighted sum of each asset's NORMALIZED CIV series
        on a common daily (business-day) calendar.

        Two corrections vs. the legacy implementation:
        1. Each asset's CIV is rebased to 1.0 at the common start date before
           weighting. Without this, an asset with raw NAV ~2500 (e.g. PPF)
           dominates one with NAV ~18 (an MF) regardless of intended weight.
        2. Assets sampled at different frequencies (monthly gold, monthly
           PPF, daily MFs) are reindexed onto a common business-day calendar
           and forward-filled before joining. A plain inner-join collapses
           the portfolio CIV to the *intersection* of all dates — effectively
           monthly when gold or PPF is present — which then gets annualized
           by ``sqrt(252)`` downstream, inflating volatility 10×.
        """
        if not self.assets:
            return TimeseriesCIV(pd.Series(dtype=float))

        raw_series = {
            name: asset.civ.value_series().sort_index() for name, asset in self.assets.items()
        }

        # Common window = latest start across assets, earliest end across assets.
        start = max(s.index.min() for s in raw_series.values())
        end = min(s.index.max() for s in raw_series.values())
        if start > end:
            return TimeseriesCIV(pd.Series(dtype=float))

        calendar = pd.bdate_range(start, end)

        reindexed = pd.DataFrame(
            {name: s.reindex(calendar, method="ffill") for name, s in raw_series.items()}
        ).dropna(how="any")

        if reindexed.empty:
            return TimeseriesCIV(pd.Series(dtype=float))

        normalized = reindexed.divide(reindexed.iloc[0])
        weights = pd.Series(self.weights).reindex(normalized.columns).fillna(0.0)
        combined = normalized.mul(weights, axis=1).sum(axis=1)
        combined.name = "value"
        return TimeseriesCIV(combined)


def from_multiple_nav_series(
    nav_dict: dict[str, pd.Series | None],
    weights: dict[str, float] | None = None
) -> PortfolioTimeseries:
    """
    Convert a dict of raw NAV series into a PortfolioTimeseries instance.

    Parameters:
        nav_dict (dict): {"fund_name": pd.Series or None}
        weights (dict): Optional weights for each asset

    Returns:
        PortfolioTimeseries: The constructed portfolio
    """
    assets: dict[str, AssetTimeseries] = {}
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


