"""Collect Value Research Online (VRO) published trailing returns.

Used to validate our own metrics against an external authority (see the
VRO-parity wire test). VRO sits behind a Cloudflare JS challenge, so — exactly
like the niftyindices scraper in ``loaders.data_update`` — the only fetch path
is a stealth Chromium browser (the optional ``browser`` extra). Once the
challenge clears, VRO's React app talks to same-origin JSON APIs; we issue the
``peer-comparison-returns`` request from *inside* the page so the session
cookies ride along automatically.

That endpoint returns the fund alongside its peers for one period; the fund's
own **annualised** trailing return (percent) is the ``returns`` entry whose
``plan_id`` matches the requested ``fund_id``:

    GET /api/funds/peer-comparison-returns/?fund_id=15841&period=5Y
    -> {"returns": [{"plan_id": "15841", "returns": "14.08"}, ...peers], ...}

**Risk ratios** (Mean / Standard Deviation / Sharpe / Sortino / Beta / Alpha)
are *not* on the overview page and *not* under ``/api/funds/*`` — the prior pass
was right that the overview API lacks them. The fund-page bundle
(``script-v2.../funds/29/...js``, function ``risk_ratio_tab_ajax``) loads them
lazily, on opening the Risk tab, from a different route that returns an **HTML
fragment** (a fund-vs-peers table), not JSON:

    GET /funds/risk-ratios-tab-data/?fund_name=15841&fund_name1=..&..&fund_name4=..&lang=en
    -> <table> fund + four peers, one row of ratios each </table>

The four peers are exactly the cohort ``peer-comparison-returns`` already
returns, so a single stealth session collects both families (see
:func:`fetch_vro_metrics`). Both endpoints are Cloudflare-challenged, so both
must be fetched from inside the cleared page.

The parse steps are pure functions (offline-unit-tested against fixtures); only
the fetch needs a browser + network.
"""

from __future__ import annotations

import csv
import json
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import DateOffset
from webgrab import browser as wg_browser

from portfolioanalyzer import metrics

# Real desktop-Chrome User-Agent presented by the stealth context (mirrors the
# niftyindices fetcher's authentic-fingerprint approach).
VRO_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Periods VRO exposes that are directly comparable to an annualised CAGR.
VRO_PERIODS: tuple[str, ...] = ("1Y", "3Y", "5Y")

# VRO publishes its risk ratios on a trailing 3-year, monthly basis. Mean / Std
# Dev / Sharpe / Sortino appear for every fund; Beta / Alpha only when the fund
# has a comparable benchmark (absent for e.g. a US-equity FoF or some debt
# funds), so they are optional.
VRO_RISK_PERIOD_YEARS = 3
VRO_RISK_REQUIRED_KEYS: tuple[str, ...] = ("mean", "std_dev", "sharpe", "sortino")
VRO_RISK_KEYS: tuple[str, ...] = (*VRO_RISK_REQUIRED_KEYS, "beta", "alpha")

# Annual risk-free rate VRO appears to assume in its Sharpe/Sortino. Back-solved
# from a captured fragment (ICICI Pru Large Cap Dir 3Y): Sharpe 0.64 = (Mean
# 14.51 - Rf) / Std 13.51 ⇒ Rf ≈ 5.9%. Used as the default for our matched
# computation; refined during live reconciliation across the mapped funds.
# VRO does not publish its assumed risk-free, so this is back-solved from VRO's
# own Mean / Std Dev / Sharpe triple -- see implied_vro_risk_free().
#
# Measured 2026-09-07 across four tracked funds: 5.62, 5.67, 5.68, 5.69 -> 5.67%.
# It was 5.90% (derived June 2026) and had drifted. That drift did NOT look like a
# stale constant: because dSharpe = dRf / StdDev, a fixed 0.23pp error surfaced as
# 0.17 on the corporate bond fund (StdDev 1.34) and 0.01 on the US equity FoF
# (StdDev 17.94). The steadiest fund in the set was the loudest symptom, which is
# exactly the wrong place to start looking.
#
# tests/unit/test_vro.py guards this constant directly, so the next drift fails
# saying "the risk-free has drifted" rather than "Sharpe is wrong for one fund".
VRO_RISK_FREE_ANNUAL = 0.0567


def implied_vro_risk_free(risk: dict) -> float:
    """VRO's implied annual risk-free (percent), from its own published triple.

    Sharpe = (mean - rf) / std_dev, so rf = mean - sharpe * std_dev. Making this
    a function rather than a hand-derivation is the point: a constant back-solved
    once by hand goes stale silently, and the symptom appears somewhere else.
    """
    mean, std, sharpe = risk["mean"], risk["std_dev"], risk["sharpe"]
    if not std:
        raise ValueError("std_dev is zero; the risk-free cannot be back-solved")
    return mean - sharpe * std

# In-page fetch used inside the (Cloudflare-cleared) page so same-origin session
# cookies ride along. Returns the raw response text; raises on non-2xx.
# In-page fetch JS comes from webgrab.browser.fetch_js, which JSON-encodes the
# URL, headers and body rather than interpolating them, and throws inside the page
# on a non-OK status. That last part is load-bearing here: a fetch that ignores
# r.ok hands back Cloudflare's challenge page as if it were fund data.
_VRO_XHR_HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}

# __file__ is .../portfolioanalyzer/loaders/vro.py; data/ is at the repo root, two
# levels above the package dir, i.e. parents[2] from here.
_VRO_FUNDS_CSV = Path(__file__).resolve().parents[2] / "data" / "funds" / "vro_funds.csv"


@dataclass(frozen=True)
class VROFund:
    """One row of the mfapi ↔ VRO fund mapping (``data/funds/vro_funds.csv``)."""

    mfapi_code: str
    vro_plan_id: str
    vro_slug: str
    isin: str
    name: str
    # The fund's stated benchmark (TRI), as a human label.
    benchmark: str = ""
    # The niftyindices index name whose TRI we can actually fetch for this
    # benchmark (see :func:`fetch_benchmark_tri`). Set only where the benchmark
    # is on niftyindices' free feed — currently NIFTY 100 (ICICI Bluechip). Empty
    # for benchmarks not on that feed (the tiered/hybrid debt indices) or out of
    # scope (Russell 3000 Growth for the US FoF); those funds' Beta/Alpha stay
    # unasserted. See the probe under ``scripts/probe_niftyindices_*.py``.
    benchmark_index: str = ""

    @property
    def mfapi_url(self) -> str:
        return f"https://api.mfapi.in/mf/{self.mfapi_code}"


def load_vro_fund_map(path: Path | None = None) -> list[VROFund]:
    """Load the mfapi ↔ VRO fund mapping from ``data/funds/vro_funds.csv``."""
    path = path or _VRO_FUNDS_CSV
    with path.open(newline="") as f:
        return [
            VROFund(
                mfapi_code=row["mfapi_code"].strip(),
                vro_plan_id=row["vro_plan_id"].strip(),
                vro_slug=row["vro_slug"].strip(),
                isin=row["isin"].strip(),
                name=row["name"].strip(),
                benchmark=(row.get("benchmark") or "").strip(),
                benchmark_index=(row.get("benchmark_index") or "").strip(),
            )
            for row in csv.DictReader(f)
        ]


def vro_page_url(plan_id: str, slug: str) -> str:
    return f"https://www.valueresearchonline.com/funds/{plan_id}/{slug}/"


def vro_returns_api(plan_id: str, period: str) -> str:
    return (
        "https://www.valueresearchonline.com/api/funds/peer-comparison-returns/"
        f"?fund_id={plan_id}&period={period}"
    )


def vro_risk_ratios_api(fund_name: str, peers: tuple[str, ...] = (), lang: str = "en") -> str:
    """URL for VRO's risk-ratios tab fragment.

    Discovered from the fund-page bundle (``risk_ratio_tab_ajax``): a GET to
    ``/funds/risk-ratios-tab-data/`` whose ``fund_name`` is the fund's **short
    name** (e.g. ``"ICICI Pru Large Cap Dir"``, read from the page's
    ``#fund_name`` hidden input — *not* the numeric plan id), with up to four
    peers as ``fund_name1..4``. Unlike the JSON return APIs it serves an **HTML
    fragment**; with no peers it returns just the target fund's row, which is all
    :func:`parse_risk_ratios` needs.
    """
    from urllib.parse import quote

    params = [f"fund_name={quote(fund_name)}"]
    params += [f"fund_name{i}={quote(p)}" for i, p in enumerate(peers[:4], start=1)]
    params.append(f"lang={lang or 'en'}")
    return "https://www.valueresearchonline.com/funds/risk-ratios-tab-data/?" + "&".join(params)


def parse_peer_comparison_returns(json_text: str, plan_id: str) -> float:
    """Extract a fund's own annualised trailing return (percent) from a
    ``peer-comparison-returns`` payload.

    Args:
        json_text: Raw JSON body from the endpoint.
        plan_id: VRO plan id of the fund of interest.

    Returns:
        The annualised return as a percent (e.g. ``14.08``).

    Raises:
        ValueError: If the payload is malformed or the plan id is absent.
    """
    try:
        payload = json.loads(json_text)
        rows = payload["returns"]
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"malformed peer-comparison-returns payload: {e}") from e

    for row in rows:
        if str(row.get("plan_id")) == str(plan_id):
            return float(row["returns"])
    raise ValueError(f"plan_id {plan_id} not found in peer-comparison returns")


# Maps the risk-ratios fragment's row labels (lower-cased, sans "(%)") to our
# canonical metric keys.
_RISK_LABEL_TO_KEY = {
    "mean": "mean",
    "std dev": "std_dev",
    "sharpe": "sharpe",
    "sortino": "sortino",
    "beta": "beta",
    "alpha": "alpha",
}


def parse_risk_ratios(html_text: str) -> dict[str, float]:
    """Extract a fund's risk ratios from the ``risk-ratios-tab-data`` fragment.

    The fragment is a small table — one row per measure, each with a label cell
    (``.first-col-custom-width``: ``Mean (%)`` / ``Std Dev (%)`` / ``Sharpe`` /
    ``Sortino`` / ``Beta`` / ``Alpha``) and a value cell (``.mono-font``). When
    fetched with no peers it holds exactly the target fund's column. Beta / Alpha
    rows are present only for funds with a comparable benchmark.

    Returns:
        ``{metric_key: value}`` covering at least :data:`VRO_RISK_REQUIRED_KEYS`,
        plus ``beta`` / ``alpha`` when published. Mean / Std Dev / Alpha are
        percent; Sharpe / Sortino / Beta are unitless.

    Raises:
        ValueError: If the fragment is unparseable or a *required* measure is
            missing.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_text, "html.parser")
    out: dict[str, float] = {}
    for tr in soup.select("tr"):
        label_el = tr.select_one(".first-col-custom-width")
        value_el = tr.select_one(".mono-font")
        if label_el is None or value_el is None:
            continue
        label = label_el.get_text(strip=True).lower().split("(")[0].strip()
        key = _RISK_LABEL_TO_KEY.get(label)
        if key is None:
            continue
        try:
            out[key] = float(value_el.get_text(strip=True))
        except ValueError:
            continue

    missing = [k for k in VRO_RISK_REQUIRED_KEYS if k not in out]
    if missing:
        raise ValueError(
            f"risk-ratios fragment missing required {missing} (parsed {sorted(out)}); "
            "the fragment shape may have changed"
        )
    return out


def trailing_cagr_pct(
    nav: pd.Series, years: int, as_of: pd.Timestamp | None = None
) -> float:
    """Our point-to-point trailing CAGR (percent), the methodology that matches
    VRO's published trailing returns.

    Reconciliation against VRO (2026-06-18) showed VRO uses point-to-point
    *daily* NAVs — latest NAV to exactly ``years`` prior — not month-end
    sampling (month-end ran ~0.5–0.8pp low). So this trims the NAV series to
    ``[anchor - years, anchor]`` and takes the plain endpoint CAGR; it agrees
    with VRO to ≤0.25pp on the funds tested.

    Args:
        nav: Daily NAV series indexed by date.
        years: Trailing window length in years.
        as_of: Anchor date; defaults to the latest NAV date.
    """
    anchor = as_of if as_of is not None else nav.index.max()
    start = anchor - DateOffset(years=years)
    window = nav[(nav.index >= start) & (nav.index <= anchor)]
    if len(window) < 2:
        raise ValueError(f"too few NAVs in the trailing {years}Y window to compute CAGR")
    return metrics.cagr(window) * 100.0


def _trailing_monthly_returns(
    nav: pd.Series, years: int, as_of: pd.Timestamp | None = None
) -> pd.Series:
    """Month-end simple returns over the trailing ``years`` window.

    VRO computes its risk ratios on a trailing-3-year, *monthly* basis and
    publishes them *as of the last complete month-end* — so we anchor on that
    month-end (stepping back from a mid-month latest NAV), resample the daily NAV
    to each calendar month's last observation, and take period-over-period
    returns (~36 points for a 3Y window). Anchoring to month-end rather than the
    latest daily NAV matters: a mid-month anchor shifts the 36-month window and
    biases Mean by a few tenths of a point.
    """
    anchor = as_of if as_of is not None else nav.index.max()
    # Last complete month-end on or before the anchor.
    month_end = anchor + pd.offsets.MonthEnd(0)
    if month_end > anchor:
        month_end = anchor - pd.offsets.MonthEnd(1)
    start = month_end - DateOffset(years=years)
    window = nav[(nav.index >= start) & (nav.index <= month_end)]
    monthly = window.resample("ME").last().dropna()
    return monthly.pct_change().dropna()


def trailing_risk_ratios(
    nav: pd.Series,
    years: int = VRO_RISK_PERIOD_YEARS,
    *,
    risk_free_annual: float = VRO_RISK_FREE_ANNUAL,
    benchmark_nav: pd.Series | None = None,
    as_of: pd.Timestamp | None = None,
) -> dict[str, float]:
    """Our risk ratios on VRO's methodology (trailing 3Y, monthly), for parity.

    Mirrors VRO's published Mean / Standard Deviation / Sharpe / Sortino (and,
    when a benchmark series is supplied, CAPM Beta / Alpha) by delegating to the
    pure ``metrics`` layer with ``periods_per_year=12``. Mean, Std Dev and Alpha
    come back as **percent** (annualised); Sharpe / Sortino / Beta are unitless.

    Args:
        nav: Daily NAV series indexed by date.
        years: Trailing window length (VRO uses 3).
        risk_free_annual: Annual risk-free rate as a fraction (e.g. ``0.06``);
            converted to a per-month rate for the excess-return measures. The
            exact value VRO assumes is pinned during reconciliation.
        benchmark_nav: Daily benchmark NAV/TRI; required only for Beta/Alpha.
        as_of: Anchor date; defaults to the latest NAV date.
    """
    rets = _trailing_monthly_returns(nav, years, as_of)
    if len(rets) < 12:
        raise ValueError(f"too few monthly returns in the trailing {years}Y window")
    rf_monthly = risk_free_annual / 12.0

    out: dict[str, float] = {
        "mean": float(rets.mean() * 12 * 100.0),
        "std_dev": metrics.volatility(rets, periods_per_year=12) * 100.0,
        "sharpe": metrics.sharpe(rets, risk_free_rate=rf_monthly, periods_per_year=12),
        "sortino": metrics.sortino(rets, risk_free_rate=rf_monthly, periods_per_year=12),
    }
    if benchmark_nav is not None:
        brets = _trailing_monthly_returns(benchmark_nav, years, as_of)
        out["beta"] = metrics.beta_capm(rets, brets, risk_free_rate=rf_monthly)
        out["alpha"] = (
            metrics.alpha_capm(rets, brets, risk_free_rate=rf_monthly, periods_per_year=12)
            * 100.0
        )
    return out


def fetch_benchmark_tri(
    index_name: str, *, start: str = "01-Jan-2007", end: str | None = None
) -> pd.Series:  # pragma: no cover - requires a real browser + network
    """Fetch a benchmark's Total-Returns-Index from niftyindices as a daily Series.

    Thin wrapper over :func:`loaders.data_update.fetch_niftyindices_tri` (same
    stealth-browser path, same ``browser`` extra) that returns the ``value``
    column as a date-indexed Series — the shape ``trailing_risk_ratios``'
    ``benchmark_nav`` expects. Used to source Beta/Alpha for funds whose stated
    benchmark is on niftyindices' feed (currently only NIFTY 100).
    """
    from portfolioanalyzer.loaders.data_update import fetch_niftyindices_tri

    frame = fetch_niftyindices_tri(index_name=index_name, start=start, end=end)
    return frame["value"].rename(index_name)


class VROFundPageGone(ValueError):
    """VRO answered 410/404 for a fund page: the stored slug is stale.

    A named type so a renamed fund is not mistaken for a parser regression.
    """


def browser_extra_available() -> bool:
    """True if the optional ``browser`` extra (playwright + stealth) is installed.

    Delegates to ``webgrab.browser.available()`` so there is one implementation
    of the check; ``webgrab.browser.require_available()`` raises a message naming
    the install steps, including that the Chromium download is separate.
    """
    return wg_browser.available()


@dataclass(frozen=True)
class VROMetrics:
    """VRO's published parity-comparable metrics for one fund.

    ``returns`` maps each period (``"3Y"``…) to the annualised trailing return
    (percent); ``risk`` maps each :data:`VRO_RISK_KEYS` name to its value
    (Mean / Std Dev / Alpha in percent, Sharpe / Sortino / Beta unitless).
    """

    plan_id: str
    returns: dict[str, float] = field(default_factory=dict)
    risk: dict[str, float] = field(default_factory=dict)


# In-page XHR with a hard abort, used for the Cloudflare-sensitive /funds/ route
# (a plain in-page fetch there can hang behind the challenge with no timeout).
@contextmanager
def _vro_session(
    plan_id: str, slug: str, *, timeout: int = 45, headless: bool = True
):  # pragma: no cover - requires a real browser + network
    """Open one stealth Chromium session with the Cloudflare challenge cleared.

    Yields the Playwright ``page`` sitting on the fund's overview page (cookies
    minted). Clearing Cloudflare is the expensive step, so callers chain every
    VRO request for a fund through a single session.

    The stealth setup -- launch, context with a desktop UA / 1920x1080 /
    en-IN / Asia-Kolkata, and a randomised settle -- is
    ``webgrab.browser.session``. It was duplicated here, in data_update's
    niftyindices fetcher, and in six probe scripts; that duplication was the
    argument for extracting it.
    """
    with wg_browser.session(
        vro_page_url(plan_id, slug),
        headless=headless,
        timeout=timeout * 1000,
        settle=(3.0, 5.0),
        user_agent=VRO_UA,
        locale="en-IN",
        timezone="Asia/Kolkata",
    ) as page:
        yield page


def _fetch_returns_texts(page, plan_id: str, periods: tuple[str, ...]) -> dict[str, str]:
    # pragma: no cover - requires a real browser + network
    """Raw ``peer-comparison-returns`` JSON per period via same-origin XHR (the
    /api/ routes are Cloudflare-exempt, so a plain in-page fetch suffices)."""
    return {
        period: page.evaluate(
            wg_browser.fetch_js(vro_returns_api(plan_id, period),
                                headers=_VRO_XHR_HEADERS))
        for period in periods
    }


def _fetch_risk_fragment(page, *, timeout: int = 45) -> str:
    # pragma: no cover - requires a real browser + network
    """Fetch the risk-ratios HTML fragment for the fund the page is showing.

    The /funds/ route is fully Cloudflare-challenged and keys on the fund's short
    name (read from ``#fund_name``). A bare navigation passes the challenge but
    the route only returns real data to an XHR; an XHR straight from the fund
    page hangs behind the challenge. So: read the name, navigate the risk URL
    once to clear Cloudflare for that path, then issue the XHR (now cleared *and*
    XHR-flagged).
    """
    info = page.evaluate(
        """() => {
            const v = id => (document.getElementById(id) || {}).value || '';
            return {
                fund: v('fund_name'),
                peers: [1, 2, 3, 4].map(i => v('peer-fund-search' + i)).filter(Boolean),
                lang: v('curr_lang') || 'en',
            };
        }"""
    )
    if not info["fund"]:
        # Distinguish "the fund page is gone" from "the page shape changed".
        # Found 2026-09-07: ICICI Bluechip was renamed to Large Cap, so its old
        # slug returned HTTP 410 and this surfaced as an unreadable #fund_name --
        # a message that sends you looking at the selector rather than the URL.
        # A renamed fund keeps its plan_id; only the slug moves.
        title = ""
        with suppress(Exception):
            title = page.title() or ""
        if "410" in title or "404" in title or "error" in title.lower():
            raise VROFundPageGone(
                f"VRO returned {title!r} for this fund page: the slug is stale, most "
                f"likely because the fund was renamed (the plan_id usually survives). "
                f"Re-identify it and update data/funds/vro_funds.csv."
            )
        raise ValueError("could not read the fund short name (#fund_name) from the page")
    url = vro_risk_ratios_api(info["fund"], tuple(info["peers"]), info["lang"])

    # We only need Cloudflare cleared for this path; the XHR below carries the
    # real request, so a navigation timeout on the (non-XHR) stub is harmless.
    with suppress(Exception):
        page.goto(url, wait_until="commit", timeout=timeout * 1000)
    for _ in range(8):
        page.wait_for_timeout(2000)
        body = page.content()
        if "Just a moment" not in body and "challenge" not in body.lower():
            break
    return page.evaluate(wg_browser.fetch_js(url, headers=_VRO_XHR_HEADERS, timeout_ms=20000))


def fetch_vro_metrics(
    plan_id: str,
    slug: str,
    periods: tuple[str, ...] = VRO_PERIODS,
    *,
    timeout: int = 45,
    headless: bool = True,
) -> VROMetrics:  # pragma: no cover - requires a real browser + network
    """All of VRO's parity-comparable metrics for a fund, in one stealth session.

    Clears Cloudflare once, then collects (1) ``peer-comparison-returns`` per
    period and (2) the ``risk-ratios-tab-data`` fragment. Returns must be fetched
    before the risk step, which navigates away from the fund page.
    """
    with _vro_session(plan_id, slug, timeout=timeout, headless=headless) as page:
        return_texts = _fetch_returns_texts(page, plan_id, periods)
        risk_html = _fetch_risk_fragment(page, timeout=timeout)

    returns = {p: parse_peer_comparison_returns(t, plan_id) for p, t in return_texts.items()}
    risk = parse_risk_ratios(risk_html)
    return VROMetrics(plan_id=str(plan_id), returns=returns, risk=risk)
