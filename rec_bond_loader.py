"""Build a REC tax-free bond cumulative-gain series from a TOML spec.

REC bonds are fixed-coupon long-duration instruments. We model them as
a constant-rate variable-bond series over a 25-year-plus horizon — the
same machinery used for SCSS. The coupon comes from the portfolio
TOML's ``[rec_bond]`` section (``coupon`` field, in percent).

Extracted from main.py:97-105, which had a hardcoded 5.25% that
silently ignored the TOML's coupon value (e.g. ``port-everything.toml``
specifies ``coupon = 5.0``).
"""

from __future__ import annotations

import pandas as pd

from bond_calculators import calculate_variable_bond_cumulative_gain

DEFAULT_COUPON = 5.25
DEFAULT_START = "2000-01-01"


def load_rec_bond_series(spec: dict) -> pd.DataFrame:
    """Build the cumulative-gain DataFrame for a REC bond.

    Args:
        spec: The ``[rec_bond]`` mapping from the portfolio TOML. Reads
            ``coupon`` (percent) if present; otherwise uses
            ``DEFAULT_COUPON``.

    Returns:
        DataFrame with a daily DatetimeIndex from 2000-01-01 to today.
    """
    coupon = spec.get("coupon", DEFAULT_COUPON)
    if not isinstance(coupon, (int, float)):
        raise ValueError(f"REC bond coupon must be numeric; got {coupon!r}")

    rates = pd.DataFrame(
        {"rate": [float(coupon)]},
        index=pd.date_range(DEFAULT_START, pd.Timestamp.today(), freq="D"),
    )
    return calculate_variable_bond_cumulative_gain(rates, rates.index.min())
