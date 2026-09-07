"""Fetch mutual-fund NAV history from the mfapi.in JSON endpoint.

The mfapi.in `data` array carries entries like ``{"date": "dd-mm-yyyy",
"nav": "123.4567"}``. This module turns that into a DataFrame with a
sorted ``DatetimeIndex`` named ``date`` and a single ``nav`` column of
floats, and raises ``RuntimeError`` when the fetch cannot be completed.

Fetching goes through the shared ``webgrab`` client. Two behaviours changed
when it did, both deliberate:

  BACKOFF. This retried ten times in a tight loop with no pause between
  attempts. That is not resilience, it is a burst -- the shape of traffic most
  likely to get a client rate-limited or blocked, and it turned one unreachable
  host into ten immediate hits. Attempts now widen the gap between tries, and
  the default is three attempts rather than ten.

  A MALFORMED PAYLOAD IS NOT RETRIED. Previously a response whose ``data`` key
  was missing or empty was retried like a network error, so a fund that returns
  a valid-but-empty history cost ten round trips before failing. A payload that
  parsed is an answer; asking again does not change it.

Extracted from ``data_loader.fetch_navs_of_mutual_fund``; the latter
remains as a thin re-export for backward compatibility.
"""

from __future__ import annotations

import logging

import pandas as pd
from webgrab import http, parse

logger = logging.getLogger(__name__)


def fetch_navs(url: str, retries: int = 3, timeout: int = 20,
               backoff: float = 1.0) -> pd.DataFrame:
    """Fetch and parse a mutual-fund NAV series.

    Args:
        url: The mfapi.in endpoint for the fund.
        retries: Total attempts (not attempts *after* the first), preserving
            this function's long-standing contract.
        timeout: Per-request timeout in seconds.
        backoff: Seconds for the first pause between attempts; each subsequent
            pause widens. Pass 0 in tests to keep them fast.

    Returns:
        DataFrame with a sorted DatetimeIndex named 'date' and a single
        float 'nav' column.

    Raises:
        RuntimeError: If the fetch fails on every attempt, or the response is
            not JSON, or it carries no NAV history.
    """
    try:
        text = http.get(url, retries=max(0, retries - 1), timeout=timeout,
                        backoff=backoff)
    except http.FetchError as e:
        raise RuntimeError(
            f"Failed to fetch NAV data from {url} after {retries} retries"
        ) from e

    try:
        payload = parse.json_body(text)
    except parse.ParseError as e:
        raise RuntimeError(f"Non-JSON response from {url}: {e}") from e

    data = payload.get("data")
    if not data:
        raise RuntimeError(f"'data' key missing or empty in API response from {url}")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["nav"] = df["nav"].astype(float)
    return df.set_index("date").sort_index()
