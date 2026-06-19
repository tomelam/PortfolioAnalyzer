"""Unit tests for ``data_loader`` helpers and validation branches.

Targets the previously-uncovered surface: config/portfolio TOML loading,
``load_portfolio_details`` validation errors (funds / PPF / gold / SCSS /
REC — SGB is covered in test_sgb_portfolio_schema.py), ``extract_weights``,
``validate_allocations``, the replay/save paths of ``fetch_portfolio_civs``,
and the small numeric helpers.
"""

from __future__ import annotations

import pandas as pd
import pytest
import toml

from data_loader import (
    _fund_slug,
    extract_weights,
    fetch_portfolio_civs,
    get_benchmark_gain_daily,
    load_config_toml,
    load_portfolio_details,
    load_portfolio_toml,
    validate_allocations,
)
from timeseries.returns import TimeseriesReturn


def _load(mocker, d):
    """Run load_portfolio_details on an in-memory dict via mocked toml.load."""
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("toml.load", return_value=d)
    return load_portfolio_details("x.toml")


def _valid_fund(name="Fund A", allocation=1.0):
    return {
        "name": name,
        "url": "https://api.mfapi.in/mf/1",
        "allocation": allocation,
        "asset_allocation": {
            "equity": 100.0,
            "debt": 0.0,
            "real_estate": 0.0,
            "commodities": 0.0,
            "cash": 0.0,
        },
    }


# --- config / portfolio TOML loaders --------------------------------------

def test_load_config_toml_missing_returns_empty(mocker):
    mocker.patch("os.path.exists", return_value=False)
    assert load_config_toml("nope.toml") == {}


def test_load_config_toml_present_loads(mocker):
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("toml.load", return_value={"quiet": True})
    assert load_config_toml("c.toml") == {"quiet": True}


def test_load_portfolio_toml_missing_raises(mocker):
    mocker.patch("os.path.exists", return_value=False)
    with pytest.raises(FileNotFoundError, match="Portfolio file not found"):
        load_portfolio_toml("nope.toml")


def test_load_portfolio_details_file_not_found(mocker):
    mocker.patch("os.path.exists", return_value=False)
    with pytest.raises(FileNotFoundError, match="File not found"):
        load_portfolio_details("nope.toml")


def test_load_portfolio_details_bad_toml(mocker):
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("toml.load", side_effect=toml.TomlDecodeError("bad", "", 0))
    with pytest.raises(ValueError, match="Invalid TOML format"):
        load_portfolio_details("x.toml")


# --- load_portfolio_details validation branches ---------------------------

def test_missing_label(mocker):
    with pytest.raises(ValueError, match="Missing required top-level key: 'label'"):
        _load(mocker, {"funds": [_valid_fund()]})


def test_no_assets(mocker):
    with pytest.raises(ValueError, match="specifies no assets"):
        _load(mocker, {"label": "x"})


def test_funds_not_a_list(mocker):
    with pytest.raises(ValueError, match="'funds' must be a list"):
        _load(mocker, {"label": "x", "funds": "nope"})


def test_fund_missing_required_key(mocker):
    fund = _valid_fund()
    del fund["url"]
    with pytest.raises(ValueError, match=r"Missing required key 'url'"):
        _load(mocker, {"label": "x", "funds": [fund]})


def test_fund_asset_allocation_not_dict(mocker):
    fund = _valid_fund()
    fund["asset_allocation"] = "nope"
    with pytest.raises(ValueError, match="'asset_allocation' must be a dictionary"):
        _load(mocker, {"label": "x", "funds": [fund]})


def test_fund_asset_allocation_missing_key(mocker):
    fund = _valid_fund()
    del fund["asset_allocation"]["cash"]
    with pytest.raises(ValueError, match=r"Missing key in 'asset_allocation'.*'cash'"):
        _load(mocker, {"label": "x", "funds": [fund]})


def test_fund_asset_allocation_negative_value(mocker):
    fund = _valid_fund()
    fund["asset_allocation"]["equity"] = -1.0
    with pytest.raises(ValueError, match="Must be a non-negative number"):
        _load(mocker, {"label": "x", "funds": [fund]})


@pytest.mark.parametrize(
    "key,section,name",
    [
        ("ppf", {"name": "PPF"}, "PPF"),
        ("gold", {"name": "Gold"}, "Gold"),
        ("scss", {"name": "SCSS"}, "SCSS"),
        ("rec_bond", {"name": "REC"}, "REC"),
    ],
)
def test_asset_missing_allocation(mocker, key, section, name):
    with pytest.raises(ValueError, match="Missing required key 'allocation'"):
        _load(mocker, {"label": "x", key: section})


@pytest.mark.parametrize("key", ["ppf", "gold", "scss", "rec_bond"])
def test_asset_invalid_allocation(mocker, key):
    with pytest.raises(ValueError, match="(Invalid allocation|allocation)"):
        _load(mocker, {"label": "x", key: {"allocation": 1.5}})


def test_rec_bond_invalid_coupon(mocker):
    with pytest.raises(ValueError, match="Invalid coupon value"):
        _load(mocker, {"label": "x", "rec_bond": {"allocation": 1.0, "coupon": -1.0}})


def test_valid_multi_asset_portfolio_loads(mocker):
    """A fully-valid portfolio exercises the happy branches of every
    asset-type validator and validate_allocations."""
    d = {
        "label": "Everything",
        "funds": [_valid_fund(allocation=0.4)],
        "ppf": {"allocation": 0.2},
        "gold": {"allocation": 0.2},
        "scss": {"allocation": 0.1},
        "rec_bond": {"allocation": 0.1, "coupon": 5.25},
    }
    out = _load(mocker, d)
    assert out["label"] == "Everything"


# --- extract_weights -------------------------------------------------------

def test_extract_weights_funds_and_assets():
    w = extract_weights(
        {
            "funds": [_valid_fund("A", 0.5)],
            "gold": {"allocation": 0.2},
            "ppf": {"allocation": 0.1},
            "scss": {"allocation": 0.1},
            "rec_bond": {"allocation": 0.1},
        }
    )
    assert w == {"A": 0.5, "Gold": 0.2, "PPF": 0.1, "SCSS": 0.1, "REC": 0.1}


def test_extract_weights_fund_missing_allocation():
    with pytest.raises(ValueError, match="Each fund must have"):
        extract_weights({"funds": [{"name": "A"}]})


def test_extract_weights_asset_missing_allocation():
    with pytest.raises(ValueError, match="Missing 'allocation' for Gold"):
        extract_weights({"gold": {}})


def test_extract_weights_sgb_per_tranche():
    w = extract_weights({"sgb": [{"tranche_id": "2020-21", "allocation": 0.3}]})
    assert w == {"SGB 2020-21": 0.3}


def test_extract_weights_sgb_missing_keys():
    with pytest.raises(ValueError, match=r"Each \[\[sgb\]\] entry must have"):
        extract_weights({"sgb": [{"allocation": 0.3}]})


# --- validate_allocations --------------------------------------------------

def test_validate_allocations_sums_to_one():
    validate_allocations(
        {
            "funds": [{"allocation": 0.4}],
            "ppf": {"allocation": 0.1},
            "gold": {"allocation": 0.1},
            "sgb": [{"allocation": 0.1}],
            "scss": {"allocation": 0.1},
            "rec_bond": {"allocation": 0.2},
        }
    )  # no raise


def test_validate_allocations_off_total_raises():
    with pytest.raises(ValueError, match="must sum to 1.00"):
        validate_allocations({"funds": [{"allocation": 0.4}], "gold": {"allocation": 0.2}})


# --- replay / save paths of fetch_portfolio_civs --------------------------

def test_fetch_portfolio_civs_replay_from(tmp_path):
    navs = tmp_path / "navs"
    navs.mkdir()
    (navs / "fund-a.csv").write_text("date,nav\n2020-01-01,10.0\n2020-01-02,11.0\n")
    out = fetch_portfolio_civs({"funds": [{"name": "Fund A", "url": "x"}]}, replay_from=str(tmp_path))
    assert list(out) == ["Fund A"]
    assert list(out["Fund A"]["nav"]) == [10.0, 11.0]


def test_fetch_portfolio_civs_save_replay(tmp_path, mocker):
    df = pd.DataFrame(
        {"nav": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"], name="date"),
    )
    mocker.patch("data_loader.fetch_navs_of_mutual_fund", return_value=df)
    fetch_portfolio_civs({"funds": [{"name": "Fund A", "url": "x"}]}, save_replay=str(tmp_path))
    assert (tmp_path / "navs" / "fund-a.csv").exists()


# --- small helpers ---------------------------------------------------------

def test_fund_slug():
    assert _fund_slug("ICICI Bluechip!") == "icici-bluechip"
    assert _fund_slug("---") == "fund"


def test_get_benchmark_gain_daily():
    prices = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.date_range("2020-01-01", periods=3),
        name="value",
    )
    gain = get_benchmark_gain_daily(TimeseriesReturn(prices))
    assert gain.index.name == "date"
    assert gain.iloc[0] == 0.0  # first pct_change filled with 0
    assert gain.iloc[1] == pytest.approx(0.10)
