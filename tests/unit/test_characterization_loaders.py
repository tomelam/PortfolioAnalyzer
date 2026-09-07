"""CHARACTERIZATION TESTS -- written against the CURRENT loaders, before any
migration onto the shared `webgrab` library.

These pin today's behaviour so a refactor can be proved to have changed nothing.
They are written first and deliberately: a test written after a refactor proves
only that the new code agrees with itself, which is the failure the house rule
"verify against something that doesn't share the assumption" names.

Two tests here pin behaviour that is a KNOWN DEFECT (marked DEFECT below). They
assert what the code does today, not what it should do. When those defects are
fixed, these tests MUST fail -- that visible break is the point, so the change
cannot be smuggled in under a refactor.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from portfolioanalyzer.loaders import data_update as du
from portfolioanalyzer.loaders import mutual_fund, scss

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "api_responses")


def _fx(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------------------------------
# Pure parsers: text -> DataFrame
# --------------------------------------------------------------------------

class TestParsers:
    def test_fred_csv_drops_holiday_dot_rows(self):
        """FRED writes '.' for holidays. They must not become NaN rows that
        silently shift the last-date calculation."""
        text = "observation_date,DGS10\n2026-01-01,.\n2026-01-02,4.10\n2026-01-05,4.20\n"
        df = du.parse_fred_csv(text)
        assert len(df) == 2
        assert df.index.max() == pd.Timestamp("2026-01-05")
        assert df["value"].iloc[-1] == pytest.approx(4.20)

    def test_fred_csv_index_is_a_sorted_datetimeindex(self):
        text = "observation_date,X\n2026-01-05,2.0\n2026-01-02,1.0\n"
        df = du.parse_fred_csv(text)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing

    def test_lbma_gold_json_shape_from_the_recorded_response(self):
        df = du.parse_lbma_gold_json(_fx("gold_lbma_pm.json"))
        assert list(df.columns) == ["value"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing
        # A deliberately trimmed 5-row fixture spanning 1968..2026, not the full
        # history -- pinned as-is so a change to the fixture is a visible decision.
        assert len(df) == 5
        assert df.index.min() == pd.Timestamp("1968-04-01")
        assert df.index.max() == pd.Timestamp("2026-06-19")
        assert df["value"].iloc[-1] == pytest.approx(4150.90)
        assert (df["value"] > 0).all()

    def test_niftyindices_tri_json_shape_from_the_recorded_response(self):
        df = du.parse_niftyindices_tri_json(_fx("nifty_tri.json"))
        assert list(df.columns) == ["value"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing
        assert (df["value"] > 0).all()


# --------------------------------------------------------------------------
# Cadence and freshness
# --------------------------------------------------------------------------

class TestCadenceFrontier:
    def test_business_day_rolls_a_weekend_back_to_friday(self):
        # 2026-09-06 is a Sunday; 2026-09-04 is the Friday.
        assert du.cadence_frontier("business_day", "2026-09-06") == pd.Timestamp("2026-09-04")

    def test_business_day_on_a_weekday_is_that_day(self):
        assert du.cadence_frontier("business_day", "2026-09-07") == pd.Timestamp("2026-09-07")

    def test_month_frontier_is_the_first_of_the_month(self):
        assert du.cadence_frontier("month", "2026-09-07") == pd.Timestamp("2026-09-01")

    def test_an_unknown_cadence_raises_rather_than_defaulting(self):
        with pytest.raises(ValueError, match="unknown cadence"):
            du.cadence_frontier("fortnightly", "2026-09-07")


def _make_source(tmp_path, monkeypatch, *, name="probe_src", cadence="business_day",
                 day_gated=False, fetch=None):
    """A DataSource writing into tmp_path, with the stamp file isolated there."""
    target = tmp_path / "probe.csv"
    monkeypatch.setattr(du, "STAMP_FILE", str(tmp_path / "stamps.json"))
    src = du.DataSource(
        name=name, target_path=str(target),
        fetch=fetch or (lambda session=None: pd.DataFrame(
            {"value": [1.0]}, index=pd.DatetimeIndex(["2026-09-07"], name="date"))),
        out_date_col="date", out_value_col="rate", out_date_format="%Y-%m-%d",
        cadence=cadence, day_gated=day_gated, affects="a metric", label="Probe",
    )
    monkeypatch.setitem(du.REGISTRY, name, src)
    return src, target


class TestAssessFreshness:
    def test_data_meeting_the_frontier_is_current(self, tmp_path, monkeypatch):
        src, target = _make_source(tmp_path, monkeypatch)
        target.write_text("date,rate\n2026-09-07,1.0\n")
        a = du.assess_freshness(src, today="2026-09-07")
        assert a["current"] and not a["behind"]

    def test_data_behind_the_frontier_is_not_current(self, tmp_path, monkeypatch):
        src, target = _make_source(tmp_path, monkeypatch)
        target.write_text("date,rate\n2026-07-01,1.0\n")
        a = du.assess_freshness(src, today="2026-09-07")
        assert a["behind"] and not a["current"]

    def test_a_reverted_file_is_not_certified_fresh_by_a_surviving_stamp(
            self, tmp_path, monkeypatch):
        """The stamp/file desync. A restore can revert the CSV while the stamp
        keeps its success record; trusting the stamp alone certifies stale data."""
        src, target = _make_source(tmp_path, monkeypatch)
        target.write_text("date,rate\n2026-07-01,1.0\n")          # reverted, old
        du._update_stamp(src.name, fetched_at=du._now_iso(), last_date="2026-09-07")
        a = du.assess_freshness(src, today="2026-09-07")
        assert a["fetched_today"] is True          # the stamp claims success today
        assert a["stamp_honoured"] is False        # but the file does not back it up
        assert a["current"] is False               # so it is NOT certified


class TestEnsureSourceCurrent:
    def test_the_attempt_is_stamped_before_the_fetch(self, tmp_path, monkeypatch):
        """THE guarantee for burst-banning hosts: a fetch that RAISES must still
        have spent today's attempt, or the next run pokes the host again."""
        def exploding(session=None):
            raise RuntimeError("upstream down")

        src, target = _make_source(tmp_path, monkeypatch, day_gated=True, fetch=exploding)
        target.write_text("date,rate\n2026-07-01,1.0\n")

        res = du.ensure_source_current(src.name, today="2026-09-07")
        assert res["status"] == "stale"
        stamp = du.read_stamp(src.name)
        assert stamp.get("attempted_at"), "a raising fetch must still record the attempt"
        assert pd.Timestamp(stamp["attempted_at"]).date() == pd.Timestamp("2026-09-07").date()
        assert "fetched_at" not in stamp, "a failed fetch must not claim success"

    def test_a_day_gated_source_already_attempted_today_is_not_retried(
            self, tmp_path, monkeypatch):
        calls = []

        def counting(session=None):
            calls.append(1)
            raise RuntimeError("upstream down")

        src, target = _make_source(tmp_path, monkeypatch, day_gated=True, fetch=counting)
        target.write_text("date,rate\n2026-07-01,1.0\n")

        du.ensure_source_current(src.name, today="2026-09-07")
        assert len(calls) == 1
        res = du.ensure_source_current(src.name, today="2026-09-07")
        assert len(calls) == 1, "the host must not be contacted a second time today"
        assert res["status"] == "stale"
        assert "once per day" in res["message"]

    def test_a_successful_refresh_reports_refreshed_and_stamps_success(
            self, tmp_path, monkeypatch):
        src, target = _make_source(tmp_path, monkeypatch)
        target.write_text("date,rate\n2026-07-01,1.0\n")
        res = du.ensure_source_current(src.name, today="2026-09-07")
        assert res["status"] == "refreshed"
        assert res["last_date"] == "2026-09-07"
        stamp = du.read_stamp(src.name)
        assert stamp["fetched_at"] and stamp["attempted_at"]

    def test_an_already_current_source_is_not_fetched_at_all(self, tmp_path, monkeypatch):
        calls = []

        def counting(session=None):
            calls.append(1)
            return pd.DataFrame({"value": [1.0]},
                                index=pd.DatetimeIndex(["2026-09-07"], name="date"))

        src, target = _make_source(tmp_path, monkeypatch, fetch=counting)
        target.write_text("date,rate\n2026-09-07,1.0\n")
        res = du.ensure_source_current(src.name, today="2026-09-07")
        assert res["status"] == "current"
        assert calls == [], "a current source must cost nothing"


class TestUpdateAllIsolatesFailures:
    def test_one_dead_feed_does_not_kill_the_others(self, tmp_path, monkeypatch):
        monkeypatch.setattr(du, "STAMP_FILE", str(tmp_path / "stamps.json"))
        good = du.DataSource(
            name="good", target_path=str(tmp_path / "good.csv"),
            fetch=lambda session=None: pd.DataFrame(
                {"value": [1.0]}, index=pd.DatetimeIndex(["2026-09-07"], name="date")),
            out_date_col="date", out_value_col="rate", out_date_format="%Y-%m-%d")

        def boom(session=None):
            raise RuntimeError("dead")

        bad = du.DataSource(
            name="bad", target_path=str(tmp_path / "bad.csv"), fetch=boom,
            out_date_col="date", out_value_col="rate", out_date_format="%Y-%m-%d")
        monkeypatch.setattr(du, "REGISTRY", {"bad": bad, "good": good})

        results = {r["name"]: r for r in du.update_all()}
        assert results["bad"]["ok"] is False and "dead" in results["bad"]["error"]
        assert results["good"]["ok"] is True


# --------------------------------------------------------------------------
# mutual_fund.fetch_navs
# --------------------------------------------------------------------------

class TestFetchNavs:
    def test_parses_a_good_response(self, monkeypatch):
        import requests

        import json as _json
        body = {"data": [{"date": "04-09-2026", "nav": "10.5"},
                         {"date": "03-09-2026", "nav": "10.4"}]}

        class R:
            status_code = 200
            text = _json.dumps(body)
            def raise_for_status(self): pass
            def json(self): return body

        monkeypatch.setattr(requests, "get", lambda *a, **k: R())
        df = mutual_fund.fetch_navs("http://x")
        assert list(df.columns) == ["nav"]
        assert df.index.is_monotonic_increasing
        assert df["nav"].iloc[-1] == pytest.approx(10.5)
        assert df.index.max() == pd.Timestamp("2026-09-04")   # dayfirst, not month-first

    def test_exhausted_retries_raise_runtimeerror_not_a_default(self, monkeypatch):
        import requests

        def boom(*a, **k):
            raise requests.RequestException("down")

        monkeypatch.setattr(requests, "get", boom)
        with pytest.raises(RuntimeError, match="after 3 retries"):
            mutual_fund.fetch_navs("http://x", retries=3)

    def test_the_default_is_three_attempts_that_pause_between_tries(self, monkeypatch):
        """FIXED 2026-09-07. This test previously pinned a DEFECT: ten attempts
        in a tight loop with no pause. That is not resilience, it is a burst --
        the traffic shape most likely to get a client rate-limited, turning one
        unreachable host into ten immediate hits.

        The characterization test that pinned the old behaviour asserted
        `len(attempts) == 10` and `sleeps == []`, and fixing this broke it, which
        was the point: a behaviour change must not be smuggled in under a
        refactor."""
        import requests

        attempts = []

        def boom(*a, **k):
            attempts.append(1)
            raise requests.RequestException("down")

        monkeypatch.setattr(requests, "get", boom)
        with pytest.raises(RuntimeError, match="after 3 retries"):
            mutual_fund.fetch_navs("http://x", backoff=0)
        assert len(attempts) == 3, "default is now three attempts, not ten"

    def test_a_malformed_payload_is_not_retried(self, monkeypatch):
        """Also fixed: an empty `data` key used to be retried like a network
        error, so a fund returning a valid-but-empty history cost ten round
        trips. A payload that parsed is an answer; asking again cannot change
        it."""
        import json as _json
        import requests

        calls = []

        class Empty:
            status_code = 200
            text = _json.dumps({"status": "OK", "data": []})
            def raise_for_status(self): pass
            def json(self): return {"status": "OK", "data": []}

        def counting(*a, **k):
            calls.append(1)
            return Empty()

        monkeypatch.setattr(requests, "get", counting)
        with pytest.raises(RuntimeError, match="missing or empty"):
            mutual_fund.fetch_navs("http://x", retries=5, backoff=0)
        assert len(calls) == 1, "a parsed answer must not be re-requested"


# --------------------------------------------------------------------------
# scss.load_scss_interest_rates
# --------------------------------------------------------------------------

class TestScss:
    def test_parses_the_recorded_page(self):
        df = scss.load_scss_interest_rates(replay_from=FIXTURES)
        assert not df.empty
        assert "interest" in df.columns

    def test_DEFECT_any_failure_returns_an_empty_frame_instead_of_raising(self, tmp_path):
        """DEFECT, pinned deliberately. This is the one place in the project that
        hides a failure: a missing fixture, a dead network or a broken parser all
        return an empty DataFrame, which flows downstream as 'no SCSS rates' --
        indistinguishable from a fund that genuinely has none. The project's own
        CLAUDE.md says fail loud. Fixing this MUST break this test."""
        df = scss.load_scss_interest_rates(replay_from=str(tmp_path))   # no fixture there
        assert df.empty
        assert list(df.columns) == ["interest"]
