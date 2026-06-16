"""Reference data + lookup API for Sovereign Gold Bond tranches.

Scoped to tranches purchasable from February 2020 onward — the window
the portfolio actually held. The earlier 56 tranches (Nov 2015 → Jan
2020) are intentionally out of scope.

Source of truth: ``data/sgb_tranches.csv``. The two tranches the user
holds (FY 2019-20 Series IX and FY 2020-21 Series VII) are anchored
to the actual RBI certificates; the rest are sourced from the
consolidated SGB master ledger with explicit verify-against-PRID
notes per row where the issue date is an estimate rather than
direct confirmation.

Companion to ``sgb_holdings.py``: that module's valuation engine looks
up tranche details here (``lookup_tranche``) when building the CIV
series.
"""

from __future__ import annotations

import datetime as dt
import os
from functools import lru_cache
from typing import Literal

import pandas as pd

# Earliest issue date in the reference data. Used for the "no tranches
# from before this date" invariant and as a public re-export so callers
# can document the scope decision.
PURCHASABLE_FROM = dt.date(2020, 2, 1)

_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sgb_tranches.csv")

_Status = Literal["LOCK", "PRE", "MAT"]


@lru_cache(maxsize=1)
def load_tranches() -> pd.DataFrame:
    """Return the full tranche reference DataFrame.

    Dates are parsed to ``datetime.date``; prices to ``int``; coupon
    rate to ``float``. The result is cached for the process lifetime.
    """
    df = pd.read_csv(
        _DATA_PATH,
        dtype={
            "tranche_id": str,
            "fiscal_year": str,
            "series": str,
            "series_num": int,
            "issue_price_offline": int,
            "issue_price_online": int,
            "coupon_rate_pct": float,
            "rbi_prid": str,
        },
        parse_dates=["issue_date", "maturity_date", "premature_first_date"],
        date_format="%Y-%m-%d",
    )
    # Normalize Timestamp → date for the date columns.
    for col in ("issue_date", "maturity_date", "premature_first_date"):
        df[col] = df[col].dt.date
    return df


def lookup_tranche(tranche_id: str) -> dict:
    """Return one tranche as a plain dict keyed by column name.

    Raises ``KeyError`` if the tranche isn't in the reference.
    """
    df = load_tranches()
    matches = df[df["tranche_id"] == tranche_id]
    if matches.empty:
        raise KeyError(
            f"unknown tranche {tranche_id!r}; "
            f"valid IDs include e.g. {df['tranche_id'].head(3).tolist()}"
        )
    return matches.iloc[0].to_dict()


def tranche_status(tranche_id: str, as_of: dt.date) -> _Status:
    """Lifecycle stage of a tranche as of ``as_of``.

    - ``LOCK`` — within the 5-year lock-in (no exit window yet).
    - ``PRE``  — past the 5-year anniversary, before 8-year maturity:
                 premature redemption via RBI semi-annual windows is
                 allowed. Secondary-market exit is also available.
    - ``MAT``  — past the 8-year maturity (final redemption due / done).
    """
    t = lookup_tranche(tranche_id)
    if as_of >= t["maturity_date"]:
        return "MAT"
    if as_of >= t["premature_first_date"]:
        return "PRE"
    return "LOCK"


def list_tranches(
    fiscal_year: str | None = None,
    status: _Status | None = None,
    as_of: dt.date | None = None,
) -> pd.DataFrame:
    """Filter the reference to a fiscal year and/or status.

    ``status`` filtering requires ``as_of``. Filters are AND-combined.
    Returns a copy so callers can mutate freely.
    """
    df = load_tranches().copy()
    if fiscal_year is not None:
        df = df[df["fiscal_year"] == fiscal_year]
    if status is not None:
        if as_of is None:
            raise ValueError("status filter requires as_of to be set")
        df = df[df["tranche_id"].apply(lambda t: tranche_status(t, as_of) == status)]
    return df.reset_index(drop=True)


def describe_tranche(tranche_id: str, as_of: dt.date | None = None) -> str:
    """Multi-line human-readable summary for the CLI / REPL / debug.

    The tone matches the rest of main.py's stdout report; not localized.
    """
    t = lookup_tranche(tranche_id)
    as_of = as_of or dt.date.today()
    status = tranche_status(tranche_id, as_of)
    lines = [
        f"Tranche {t['tranche_id']}  ({t['fiscal_year']} Series {t['series']})",
        f"  Issue:    {t['issue_date']:%d %b %Y} @ ₹{t['issue_price_offline']:,}/gm "
        f"(offline) / ₹{t['issue_price_online']:,}/gm (online)",
        f"  Maturity: {t['maturity_date']:%d %b %Y}",
        f"  Premature redemption window opens: {t['premature_first_date']:%d %b %Y}",
        f"  Coupon:   {t['coupon_rate_pct']:.2f}% p.a. on offline nominal value",
        f"  Status:   {status}  (as of {as_of:%Y-%m-%d})",
        f"  RBI PRID: {t['rbi_prid']}",
    ]
    return "\n".join(lines)
