"""Unit tests for the data auto-update subsystem (``loaders.data_update``).

Network is mocked here via a fake session; the one live FRED fetch is in
``test_data_update_live.py`` behind the ``network`` marker. The writer's
output is fed back through the real ``load_timeseries_csv`` loader to prove
the normalized schema is pipeline-compatible.
"""

from __future__ import annotations

import json

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

NIFTY_SAMPLE = (
    "Date,Total Returns Index\n"
    '01-Jan-2020,"12,000.50"\n'
    '02-Jan-2020,"12,050.75"\n'
    '03-Jan-2020,"11,990.10"\n'
)


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


def test_parse_niftyindices_tri_csv():
    df = du.parse_niftyindices_tri_csv(NIFTY_SAMPLE)
    assert list(df.columns) == ["value"]
    assert len(df) == 3
    # comma thousands stripped; dayfirst dates parsed + sorted
    assert df["value"].iloc[0] == pytest.approx(12000.50)
    assert df.index[0] == pd.Timestamp("2020-01-01")


def test_parse_niftyindices_no_value_column_raises():
    with pytest.raises(ValueError, match="value column"):
        du.parse_niftyindices_tri_csv("Date,Foo\n01-Jan-2020,x\n")


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
        description="test",
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
        description="test",
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
        description="ok",
    )
    bad = du.DataSource(
        name="bad",
        target_path=str(tmp_path / "bad.csv"),
        fetch=_boom,
        out_date_col="Date",
        out_value_col="price",
        out_date_format="%m/%d/%Y",
        description="down",
    )
    monkeypatch.setattr(du, "REGISTRY", {"good": good, "bad": bad})
    results = {r["name"]: r for r in du.update_all()}
    assert results["good"]["ok"] is True
    assert results["bad"]["ok"] is False
    assert "source down" in results["bad"]["error"]
    # a dead source must not block writing the healthy one
    assert (tmp_path / "good.csv").exists()
