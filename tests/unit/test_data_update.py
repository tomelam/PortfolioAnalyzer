"""Unit tests for the data auto-update subsystem (``loaders.data_update``).

Network is mocked here via a fake session; the one live FRED fetch is in
``test_data_update_live.py`` behind the ``network`` marker. The writer's
output is fed back through the real ``load_timeseries_csv`` loader to prove
the normalized schema is pipeline-compatible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from loaders import data_update as du
from loaders.benchmark import load_timeseries_csv

FRED_SAMPLE = (
    "observation_date,INDIRLTLT01STM\n"
    "2011-12-01,8.56\n"
    "2012-01-01,8.24\n"
    "2012-02-01,.\n"  # FRED encodes missing as '.'
    "2012-03-01,8.37\n"
)

# Authentic niftyindices TRI response envelope captured 2026-06 (3 rows).
NIFTY_TRI_FIXTURE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "api_responses" / "nifty_tri.json"
).read_text()


class _FakeResp:
    def __init__(self, text, status=200):
        self.text = text
        self._status = status

    def raise_for_status(self):
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")


class _FakeSession:
    """Minimal requests-like session returning canned text per call."""

    def __init__(self, text, status=200):
        self._text = text
        self._status = status
        self.calls = []

    def get(self, url, timeout=None, headers=None):
        self.calls.append(url)
        return _FakeResp(self._text, self._status)


# --- pure parsers ----------------------------------------------------------

def test_parse_fred_csv():
    df = du.parse_fred_csv(FRED_SAMPLE)
    assert list(df.columns) == ["value"]
    assert df.index.name == "date"
    assert isinstance(df.index, pd.DatetimeIndex)
    # the '.' row is dropped, leaving 3 of 4
    assert len(df) == 3
    assert df["value"].iloc[0] == pytest.approx(8.56)
    assert df.index.is_monotonic_increasing


def test_parse_fred_csv_too_few_columns():
    with pytest.raises(ValueError, match="at least two columns"):
        du.parse_fred_csv("observation_date\n2020-01-01\n")


def test_parse_fred_csv_all_missing_raises():
    with pytest.raises(ValueError, match="zero rows"):
        du.parse_fred_csv("observation_date,X\n2020-01-01,.\n")


def test_parse_niftyindices_tri_json():
    df = du.parse_niftyindices_tri_json(NIFTY_TRI_FIXTURE)
    assert list(df.columns) == ["value"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    # "12 Jun 2026" / "35695.33" present in the captured fixture
    assert df.loc[pd.Timestamp("2026-06-12"), "value"] == pytest.approx(35695.33)


def test_parse_niftyindices_missing_envelope_raises():
    with pytest.raises(ValueError, match="'d' envelope"):
        du.parse_niftyindices_tri_json('{"notd": "x"}')


def test_parse_niftyindices_unexpected_columns_raises():
    inner = json.dumps([{"Date": "12 Jun 2026", "Foo": "1"}])
    with pytest.raises(ValueError, match="unexpected niftyindices columns"):
        du.parse_niftyindices_tri_json(json.dumps({"d": inner}))


def test_niftyindices_body_shape():
    """The POST body wraps a single-quoted cinfo JSON string with the dates."""
    body = du._niftyindices_body("NIFTY 50", "10-Jun-2026", "12-Jun-2026")
    envelope = json.loads(body)
    assert set(envelope) == {"cinfo"}
    cinfo = envelope["cinfo"]
    assert "'name':'NIFTY 50'" in cinfo
    assert "'startDate':'10-Jun-2026'" in cinfo
    assert "'endDate':'12-Jun-2026'" in cinfo


def test_fetch_niftyindices_tri_requires_browser_extra(monkeypatch):
    """Without the optional 'browser' extra the fetch raises a clear,
    install-guiding error rather than touching the network."""
    monkeypatch.setattr(du, "_playwright_available", lambda: False)
    with pytest.raises(RuntimeError, match="browser.*extra"):
        du.fetch_niftyindices_tri()


def test_fetch_niftyindices_tri_delegates_to_browser(monkeypatch):
    """When the extra is present, the fetch delegates to the stealth-browser
    path and defaults the end date to today (parsing covered separately)."""
    parsed = du.parse_niftyindices_tri_json(NIFTY_TRI_FIXTURE)
    seen = {}

    def _fake_browser(index_name, start, end, *, timeout=30, headless=True):
        seen.update(index_name=index_name, start=start, end=end)
        return parsed

    monkeypatch.setattr(du, "_playwright_available", lambda: True)
    monkeypatch.setattr(du, "_fetch_niftyindices_browser", _fake_browser)
    df = du.fetch_niftyindices_tri(start="10-Jun-2026")
    assert len(df) == 3
    assert seen["index_name"] == "NIFTY 50"
    assert seen["start"] == "10-Jun-2026"
    assert seen["end"]  # end defaulted to today's DD-Mon-YYYY


# --- network fetchers (mocked) --------------------------------------------

def test_fetch_fred_series_mocked():
    sess = _FakeSession(FRED_SAMPLE)
    df = du.fetch_fred_series("INDIRLTLT01STM", session=sess)
    assert len(df) == 3
    assert "INDIRLTLT01STM" in sess.calls[0]


# --- writer round-trips through the real loader ----------------------------

def test_write_normalized_csv_is_loader_compatible(tmp_path):
    src = du.DataSource(
        name="t",
        target_path=str(tmp_path / "rf.csv"),
        fetch=lambda session=None: None,
        out_date_col="observation_date",
        out_value_col="rate",
        out_date_format="%Y-%m-%d",
    )
    df = du.parse_fred_csv(FRED_SAMPLE)
    du.write_normalized_csv(src, df)
    # The existing benchmark loader must consume the file unchanged.
    ts = load_timeseries_csv(src.target_path, src.out_date_format, max_delay_days=None)
    assert ts.value_series().iloc[0] == pytest.approx(8.56)


# --- update_source / update_all -------------------------------------------

def test_update_source_writes_file_and_stamp(tmp_path, monkeypatch):
    monkeypatch.setattr(du, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(du, "STAMP_FILE", str(tmp_path / ".last_fetched.json"))
    target = tmp_path / "rf.csv"
    src = du.DataSource(
        name="rf",
        target_path=str(target),
        fetch=lambda session=None: du.parse_fred_csv(FRED_SAMPLE),
        out_date_col="observation_date",
        out_value_col="rate",
        out_date_format="%Y-%m-%d",
    )
    monkeypatch.setattr(du, "REGISTRY", {"rf": src})
    res = du.update_source("rf")
    assert res["ok"] is True and res["rows"] == 3
    assert target.exists()
    stamps = json.loads((tmp_path / ".last_fetched.json").read_text())
    assert "rf" in stamps and stamps["rf"]["last_date"] == "2012-03-01"


def test_update_all_isolates_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(du, "STAMP_FILE", str(tmp_path / ".last_fetched.json"))

    def _boom(session=None):
        raise RuntimeError("source down")

    good = du.DataSource(
        name="good",
        target_path=str(tmp_path / "good.csv"),
        fetch=lambda session=None: du.parse_fred_csv(FRED_SAMPLE),
        out_date_col="observation_date",
        out_value_col="rate",
        out_date_format="%Y-%m-%d",
    )
    bad = du.DataSource(
        name="bad",
        target_path=str(tmp_path / "bad.csv"),
        fetch=_boom,
        out_date_col="Date",
        out_value_col="price",
        out_date_format="%m/%d/%Y",
    )
    monkeypatch.setattr(du, "REGISTRY", {"good": good, "bad": bad})
    results = {r["name"]: r for r in du.update_all()}
    assert results["good"]["ok"] is True
    assert results["bad"]["ok"] is False
    assert "source down" in results["bad"]["error"]
    # a dead source must not block writing the healthy one
    assert (tmp_path / "good.csv").exists()


# --- cadence frontier ------------------------------------------------------

def test_cadence_frontier_business_day_rolls_back_over_weekend():
    # A Saturday's most-recent business day is the preceding Friday.
    assert du.cadence_frontier("business_day", pd.Timestamp("2026-06-13")) == pd.Timestamp(
        "2026-06-12"
    )
    # A weekday is its own frontier.
    assert du.cadence_frontier("business_day", pd.Timestamp("2026-06-15")) == pd.Timestamp(
        "2026-06-15"
    )


def test_cadence_frontier_month_is_first_of_current_month():
    assert du.cadence_frontier("month", pd.Timestamp("2026-06-18")) == pd.Timestamp("2026-06-01")


# --- cadence-driven freshness ----------------------------------------------

def _rf_source(tmp_path, fetch):
    return du.DataSource(
        name="rf",
        target_path=str(tmp_path / "INDIRLTLT01STM.csv"),
        fetch=fetch,
        out_date_col="observation_date",
        out_value_col="rate",
        out_date_format="%Y-%m-%d",
        cadence="month",
        day_gated=False,
        affects="Sharpe, Sortino, alpha",
        label="FRED India 10Y (risk-free)",
    )


def _bench_source(tmp_path, fetch):
    return du.DataSource(
        name="bench",
        target_path=str(tmp_path / "NIFTY Total Returns Historical Data.csv"),
        fetch=fetch,
        out_date_col="Date",
        out_value_col="Price",
        out_date_format="%m/%d/%Y",
        cadence="business_day",
        day_gated=True,
        affects="alpha, beta",
        label="NIFTY 50 TRI (benchmark)",
    )


def test_source_for_path_matches_by_basename():
    assert du.source_for_path("whatever/INDIRLTLT01STM.csv") is du.REGISTRY["risk_free_fred"]
    assert du.source_for_path("data/does-not-exist.csv") is None


def test_ensure_current_when_local_meets_cadence(tmp_path, monkeypatch):
    """Local data on the cadence frontier is certified current — no fetch."""
    monkeypatch.setattr(du, "STAMP_FILE", str(tmp_path / ".s.json"))
    target = tmp_path / "INDIRLTLT01STM.csv"
    target.write_text("observation_date,rate\n2026-06-01,7.0\n")  # June present
    called = {"n": 0}

    def _fetch(session=None):
        called["n"] += 1
        return du.parse_fred_csv(FRED_SAMPLE)

    monkeypatch.setattr(du, "REGISTRY", {"rf": _rf_source(tmp_path, _fetch)})
    res = du.ensure_source_current("rf", today="2026-06-18")
    assert res["status"] == "current"
    assert called["n"] == 0


def test_ensure_refreshes_when_behind(tmp_path, monkeypatch):
    """Behind the cadence frontier ⇒ fetch; a successful fetch is current."""
    monkeypatch.setattr(du, "STAMP_FILE", str(tmp_path / ".s.json"))
    target = tmp_path / "INDIRLTLT01STM.csv"
    target.write_text("observation_date,rate\n2026-04-01,7.0\n")  # April < June-01
    monkeypatch.setattr(
        du,
        "REGISTRY",
        {"rf": _rf_source(tmp_path, lambda session=None: du.parse_fred_csv(FRED_SAMPLE))},
    )
    res = du.ensure_source_current("rf", today="2026-06-18")
    assert res["status"] == "refreshed"
    stamps = json.loads((tmp_path / ".s.json").read_text())
    # both the attempt and the success are stamped
    assert "attempted_at" in stamps["rf"] and "fetched_at" in stamps["rf"]


def test_ensure_stale_when_refresh_fails(tmp_path, monkeypatch):
    """An unreachable upstream leaves the source stale but preserves data, and
    records the attempt so a day-gated source won't retry immediately."""
    monkeypatch.setattr(du, "STAMP_FILE", str(tmp_path / ".s.json"))
    target = tmp_path / "INDIRLTLT01STM.csv"
    target.write_text("observation_date,rate\n2026-04-01,7.0\n")

    def _boom(session=None):
        raise RuntimeError("upstream down")

    monkeypatch.setattr(du, "REGISTRY", {"rf": _rf_source(tmp_path, _boom)})
    res = du.ensure_source_current("rf", today="2026-06-18")
    assert res["status"] == "stale"
    assert "Could not refresh" in res["message"]
    assert target.exists()  # existing data preserved
    stamps = json.loads((tmp_path / ".s.json").read_text())
    assert "attempted_at" in stamps["rf"]  # failed attempt is recorded


def test_fetched_today_is_current_even_if_cadence_behind(tmp_path, monkeypatch):
    """The holiday/lag case: cadence says behind, but a successful fetch
    already happened today, so the source is current (latest the feed offers)
    and we don't fetch again."""
    monkeypatch.setattr(du, "STAMP_FILE", str(tmp_path / ".s.json"))
    target = tmp_path / "NIFTY Total Returns Historical Data.csv"
    target.write_text("Date,Price\n06/12/2026,100\n")  # Friday; Monday is the frontier
    (tmp_path / ".s.json").write_text(
        json.dumps({"bench": {"fetched_at": "2026-06-15T03:00:00+00:00"}})
    )
    called = {"n": 0}

    def _fetch(session=None):
        called["n"] += 1
        return du.parse_niftyindices_tri_json(NIFTY_TRI_FIXTURE)

    monkeypatch.setattr(du, "REGISTRY", {"bench": _bench_source(tmp_path, _fetch)})
    res = du.ensure_source_current("bench", today="2026-06-15")  # Monday
    assert res["status"] == "current"
    assert called["n"] == 0


def test_benchmark_once_per_day_suppresses_second_attempt(tmp_path, monkeypatch):
    """niftyindices is day-gated: a failed attempt earlier today blocks any
    retry the same day (ban-avoidance), leaving the source stale."""
    monkeypatch.setattr(du, "STAMP_FILE", str(tmp_path / ".s.json"))
    target = tmp_path / "NIFTY Total Returns Historical Data.csv"
    target.write_text("Date,Price\n01/01/2026,100\n")  # ancient, behind
    (tmp_path / ".s.json").write_text(
        json.dumps({"bench": {"attempted_at": "2026-06-17T08:00:00+00:00"}})
    )
    called = {"n": 0}

    def _fetch(session=None):
        called["n"] += 1
        return du.parse_niftyindices_tri_json(NIFTY_TRI_FIXTURE)

    monkeypatch.setattr(du, "REGISTRY", {"bench": _bench_source(tmp_path, _fetch)})
    res = du.ensure_source_current("bench", today="2026-06-17")
    assert res["status"] == "stale"
    assert called["n"] == 0  # day-gate blocked the retry


def test_fred_retries_within_day_when_not_day_gated(tmp_path, monkeypatch):
    """FRED carries no ban risk, so a failed attempt earlier today does not
    suppress a later retry the same day."""
    monkeypatch.setattr(du, "STAMP_FILE", str(tmp_path / ".s.json"))
    target = tmp_path / "INDIRLTLT01STM.csv"
    target.write_text("observation_date,rate\n2026-04-01,7.0\n")
    (tmp_path / ".s.json").write_text(
        json.dumps({"rf": {"attempted_at": "2026-06-18T08:00:00+00:00"}})
    )
    called = {"n": 0}

    def _fetch(session=None):
        called["n"] += 1
        return du.parse_fred_csv(FRED_SAMPLE)

    monkeypatch.setattr(du, "REGISTRY", {"rf": _rf_source(tmp_path, _fetch)})
    res = du.ensure_source_current("rf", today="2026-06-18")
    assert res["status"] == "refreshed"
    assert called["n"] == 1


def test_ensure_reference_data_fresh_skips_unregistered(tmp_path, monkeypatch):
    monkeypatch.setattr(du, "STAMP_FILE", str(tmp_path / ".s.json"))
    target = tmp_path / "INDIRLTLT01STM.csv"
    target.write_text("observation_date,rate\n2026-06-01,7.0\n")
    monkeypatch.setattr(
        du, "REGISTRY", {"rf": _rf_source(tmp_path, lambda session=None: None)}
    )
    results = du.ensure_reference_data_fresh(
        ["data/legacy-unregistered.csv", str(target)], today="2026-06-18"
    )
    assert [r["name"] for r in results] == ["rf"]
    assert results[0]["status"] == "current"


def test_reference_provenance_maps_paths_to_stamp(tmp_path, monkeypatch):
    monkeypatch.setattr(du, "STAMP_FILE", str(tmp_path / ".s.json"))
    (tmp_path / ".s.json").write_text(
        json.dumps(
            {
                "rf": {
                    "last_date": "2026-05-01",
                    "fetched_at": "2026-06-17T14:00:00+00:00",
                    "attempted_at": "2026-06-17T14:00:00+00:00",
                }
            }
        )
    )
    monkeypatch.setattr(du, "REGISTRY", {"rf": _rf_source(tmp_path, lambda session=None: None)})
    prov = du.reference_provenance(
        ["data/unregistered.csv", str(tmp_path / "INDIRLTLT01STM.csv")]
    )
    assert len(prov) == 1  # unregistered path skipped
    e = prov[0]
    assert e["name"] == "rf" and e["label"] == "FRED India 10Y (risk-free)"
    assert e["last_date"] == "2026-05-01"
    assert e["fetched_at"].startswith("2026-06-17")
    assert e["attempted_at"].startswith("2026-06-17")


def test_reference_provenance_missing_stamp_yields_none_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(du, "STAMP_FILE", str(tmp_path / ".s.json"))
    monkeypatch.setattr(du, "REGISTRY", {"rf": _rf_source(tmp_path, lambda session=None: None)})
    prov = du.reference_provenance([str(tmp_path / "INDIRLTLT01STM.csv")])
    assert prov[0]["last_date"] is None and prov[0]["fetched_at"] is None


def test_read_stamp_exposes_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(du, "STAMP_FILE", str(tmp_path / ".s.json"))
    (tmp_path / ".s.json").write_text(
        json.dumps({"rf": {"last_date": "2026-05-01", "fetched_at": "2026-06-17T14:00:00+00:00"}})
    )
    st = du.read_stamp("rf")
    assert st["last_date"] == "2026-05-01"
    assert st["fetched_at"].startswith("2026-06-17")
    assert du.read_stamp("missing") == {}
