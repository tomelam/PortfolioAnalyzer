"""Load Senior Citizens Savings Scheme (SCSS) interest rates.

Scrapes the historical-rate table from nsiindia.gov.in. Decomposed into
three layers so the parser is independently unit-testable:

- ``parse_scss_html`` — pure: HTML → DataFrame indexed by start date
- ``fetch_scss_html``  — network: URL → HTML text
- ``load_scss_interest_rates`` — composition with safe failure (returns
  an empty DataFrame on any error, matching prior behavior)

Extracted from ``data_loader.load_scss_interest_rates``; that name is
kept there as a thin re-export.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup

from portfolioanalyzer.utils import info

SCSS_URL = (
    "https://www.nsiindia.gov.in/(S(2xgxs555qwdlfb2p4ub03n3n))/"
    "InternalPage.aspx?Id_Pk=181"
)


def fetch_scss_html(url: str, verify_ssl: bool = False) -> str:
    """Fetch the SCSS-rate page. Disables SSL warnings (the page uses a
    self-signed-ish chain)."""
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    response = requests.get(url, verify=verify_ssl)
    response.raise_for_status()
    return response.text


def _find_target_table(soup: BeautifulSoup):
    """Find a <table>/<tbody> whose first row is YEAR | INTEREST...."""
    for container in soup.find_all(["table", "tbody"]):
        first_row = container.find("tr")
        if first_row:
            cells = first_row.find_all(["td", "th"])
            if len(cells) >= 2:
                header1 = cells[0].get_text(strip=True).lower()
                header2 = cells[1].get_text(strip=True).lower()
                if header1 == "year" and "interest" in header2:
                    return container
    return None


def _extract_rate_series(html: str) -> list[dict]:
    """Pull table rows out as a list of header-keyed dicts."""
    soup = BeautifulSoup(html, "html.parser")
    target = _find_target_table(soup)
    if not target:
        return []

    header_row = target.find("tr")
    headers = [cell.get_text(strip=True) for cell in header_row.find_all(["td", "th"])]

    rate_series = []
    for row in target.find_all("tr")[1:]:
        cells_text = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
        if len(cells_text) < len(headers):
            cells_text.extend([""] * (len(headers) - len(cells_text)))
        else:
            cells_text = cells_text[: len(headers)]
        rate_series.append(dict(zip(headers, cells_text, strict=True)))
    return rate_series


def _parse_financial_year_start(year_str: str) -> pd.Timestamp:
    """Parse the start date out of '02-08-2004 to 31-03-2012' style ranges."""
    start_str = year_str.split("to")[0].strip()
    for fmt in ("%d-%m-%Y", "%d.%m.%Y"):
        try:
            return pd.to_datetime(start_str, format=fmt)
        except Exception:
            continue
    return pd.to_datetime(start_str, dayfirst=True, errors="coerce")


def _process_rate_series(raw_series: list[dict]) -> list[dict]:
    """Turn raw header-keyed rows into ``{date, interest}`` records."""
    processed = []
    for row in raw_series:
        try:
            start_date = _parse_financial_year_start(row["YEAR"])
            interest = float(row["RATE OF INTEREST (%)"])
            processed.append({"date": start_date, "interest": interest})
        except Exception as e:
            info(f"Skipping row due to error: {e}")
    return processed


def parse_scss_html(html: str) -> pd.DataFrame:
    """Pure parser: HTML text → DataFrame indexed by financial-year start.

    Returns an empty DataFrame if the target table isn't present.
    """
    rows = _process_rate_series(_extract_rate_series(html))
    df = pd.DataFrame(rows, columns=["date", "interest"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df["interest"] = pd.to_numeric(df["interest"], errors="coerce")
    df = df.dropna(subset=["interest"])
    return df


def load_scss_interest_rates(replay_from=None, save_replay=None) -> pd.DataFrame:
    """Fetch and parse the SCSS rate history.

    Network by default. With ``replay_from`` set, the HTML is read from
    ``<replay_from>/scss_nsi.html`` instead of nsiindia.gov.in; with
    ``save_replay`` set, the fetched HTML is written there for later
    replay. On any failure (network down, parser breakage, missing
    fixture), returns an empty DataFrame — matches the prior
    swallow-and-log behavior so a missing SCSS feed doesn't crash an
    otherwise-valid portfolio run.
    """
    try:
        if replay_from is not None:
            with open(os.path.join(replay_from, "scss_nsi.html"), encoding="utf-8") as f:
                html = f.read()
        else:
            html = fetch_scss_html(SCSS_URL)
        if save_replay is not None:
            os.makedirs(save_replay, exist_ok=True)
            with open(os.path.join(save_replay, "scss_nsi.html"), "w", encoding="utf-8") as f:
                f.write(html)
        return parse_scss_html(html)
    except Exception as e:
        info(f"Error fetching SCSS rates: {e}")
        return pd.DataFrame(columns=["interest"])
