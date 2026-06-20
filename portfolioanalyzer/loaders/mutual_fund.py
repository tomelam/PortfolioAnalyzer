"""Fetch mutual-fund NAV history from the mfapi.in JSON endpoint.

The mfapi.in `data` array carries entries like ``{"date": "dd-mm-yyyy",
"nav": "123.4567"}``. This module turns that into a DataFrame with a
sorted ``DatetimeIndex`` named ``date`` and a single ``nav`` column of
floats, retrying transient request failures and raising
``RuntimeError`` once all retries are exhausted.

Extracted from ``data_loader.fetch_navs_of_mutual_fund``; the latter
remains as a thin re-export for backward compatibility.
"""

from __future__ import annotations

import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def fetch_navs(url: str, retries: int = 10, timeout: int = 20) -> pd.DataFrame:
    """Fetch and parse a mutual-fund NAV series.

    Args:
        url: The mfapi.in endpoint for the fund.
        retries: Max number of attempts before raising RuntimeError.
        timeout: Per-request timeout in seconds.

    Returns:
        DataFrame with a sorted DatetimeIndex named 'date' and a single
        float 'nav' column.

    Raises:
        RuntimeError: If all retries are exhausted, the response is
            malformed, or the data list is empty.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if "data" not in payload or not payload["data"]:
                raise KeyError(f"'data' key missing or empty in API response from {url}")
            df = pd.DataFrame(payload["data"])
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            df["nav"] = df["nav"].astype(float)
            return df.set_index("date").sort_index()
        except requests.RequestException as e:
            last_error = e
            logger.info("Request failed for %s (attempt %d/%d): %s", url, attempt + 1, retries, e)
        except (ValueError, KeyError) as e:
            last_error = e
            logger.info(
                "Data processing error for %s (attempt %d/%d): %s", url, attempt + 1, retries, e
            )
    raise RuntimeError(
        f"Failed to fetch NAV data from {url} after {retries} retries"
    ) from last_error
