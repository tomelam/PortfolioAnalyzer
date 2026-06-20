import sys

import pandas as pd

"""
# DEBUG is injected by main.py when the ‑d/‑‑debug flag is used; the fallback is `False` for unit tests
try:
    from portfolioanalyzer.main import DEBUG          # noqa: E402
except ImportError:
"""
DEBUG = False

def info(msg):
    '''Print informational messages to stderr so that structured output (stdout e.g. CSV) stays clean'''
    print(msg, file=sys.stderr)


def dbg(msg):
    """Print only if DEBUG is enabled (stderr, like info())."""
    """
    try:
        from portfolioanalyzer.main import DEBUG        # Import lazily to avoid circular refs
    except ImportError:
        DEBUG = False
    else:
        DEBUG = DEBUG
    """
    if DEBUG:
        print(msg, file=sys.stderr)


def to_cutoff_date(tag: str, as_of: pd.Timestamp | None = None) -> pd.Timestamp:
    """
    Convert a look‑back tag ('YTD', '3M', '5Y', …) to the cutoff date.

    Pass ``as_of`` to pin the evaluation date (used by --as-of for
    deterministic runs); defaults to the wall clock if omitted.
    """
    from pandas.tseries.offsets import DateOffset

    as_of = as_of if as_of is not None else pd.Timestamp.today().normalize()

    if tag == "YTD":
        return pd.Timestamp(as_of.year, 1, 1)

    number, unit = int(tag[:-1]), tag[-1].upper()
    if unit == "M":
        return as_of - DateOffset(months=number)
    if unit == "Y":
        return as_of - DateOffset(years=number)

    raise ValueError(f"Unsupported look‑back tag: {tag}")
