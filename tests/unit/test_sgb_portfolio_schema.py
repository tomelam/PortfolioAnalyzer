"""Tests for the new ``[[sgb]]`` portfolio TOML schema.

User-stated constraint (KANBAN, 2026-06-16): each SGB tranche is a
distinct portfolio investment. The previous single-entry ``[sgb]``
dict-style schema is replaced by a list of per-tranche entries:

    [[sgb]]
    tranche_id = "2019-20-IX"
    units_grams = 12
    allocation = 0.07

    [[sgb]]
    tranche_id = "2020-21-VII"
    units_grams = 6
    allocation = 0.03

These tests pin the validation + weight-extraction half of the
integration (Phase 2 of the SGB modeling refactor). main.py's actual
wire-through to ``sgb_holding_civ`` is covered by the goldens.
"""

from __future__ import annotations

import pytest

from data_loader import extract_weights, load_portfolio_details


def _write_toml(tmp_path, content):
    f = tmp_path / "port.toml"
    f.write_text(content)
    return str(f)


_MIN_VALID = """\
label = "Test"

[[funds]]
name = "Fund A"
url = "https://api.mfapi.in/mf/1"
allocation = 0.90
asset_allocation = {equity = 100.0, debt = 0.0, real_estate = 0.0, commodities = 0.0, cash = 0.0}
"""


def test_load_accepts_list_of_sgb_entries(tmp_path) -> None:
    """[[sgb]] block with two distinct tranches loads without error."""
    toml = _MIN_VALID + """
[[sgb]]
tranche_id = "2019-20-IX"
units_grams = 12
allocation = 0.05

[[sgb]]
tranche_id = "2020-21-VII"
units_grams = 6
allocation = 0.05
"""
    p = load_portfolio_details(_write_toml(tmp_path, toml))
    assert isinstance(p["sgb"], list)
    assert len(p["sgb"]) == 2
    tids = sorted(e["tranche_id"] for e in p["sgb"])
    assert tids == ["2019-20-IX", "2020-21-VII"]


def test_extract_weights_yields_one_entry_per_tranche(tmp_path) -> None:
    """Each [[sgb]] block contributes one weight keyed by tranche_id,
    so two tranches show up as two separate assets in the portfolio."""
    toml = _MIN_VALID + """
[[sgb]]
tranche_id = "2019-20-IX"
units_grams = 12
allocation = 0.07

[[sgb]]
tranche_id = "2020-21-VII"
units_grams = 6
allocation = 0.03
"""
    p = load_portfolio_details(_write_toml(tmp_path, toml))
    w = extract_weights(p)
    assert "SGB 2019-20-IX" in w
    assert "SGB 2020-21-VII" in w
    assert w["SGB 2019-20-IX"] == pytest.approx(0.07)
    assert w["SGB 2020-21-VII"] == pytest.approx(0.03)


def test_missing_tranche_id_rejected(tmp_path) -> None:
    toml = _MIN_VALID + """
[[sgb]]
units_grams = 5
allocation = 0.05
"""
    with pytest.raises(ValueError, match="tranche_id"):
        load_portfolio_details(_write_toml(tmp_path, toml))


def test_missing_units_grams_rejected(tmp_path) -> None:
    toml = _MIN_VALID + """
[[sgb]]
tranche_id = "2020-21-VII"
allocation = 0.05
"""
    with pytest.raises(ValueError, match="units_grams"):
        load_portfolio_details(_write_toml(tmp_path, toml))


def test_missing_allocation_rejected(tmp_path) -> None:
    toml = _MIN_VALID + """
[[sgb]]
tranche_id = "2020-21-VII"
units_grams = 6
"""
    with pytest.raises(ValueError, match="allocation"):
        load_portfolio_details(_write_toml(tmp_path, toml))


def test_negative_units_rejected(tmp_path) -> None:
    toml = _MIN_VALID + """
[[sgb]]
tranche_id = "2020-21-VII"
units_grams = -1
allocation = 0.05
"""
    with pytest.raises(ValueError, match="units_grams"):
        load_portfolio_details(_write_toml(tmp_path, toml))


def test_old_dict_style_sgb_rejected_with_helpful_message(tmp_path) -> None:
    """The legacy ``[sgb]`` dict block (single allocation, no tranche
    information) is no longer accepted — Phase 2 of the refactor is
    explicit that each tranche must be its own entry."""
    toml = _MIN_VALID + """
[sgb]
name = "Sovereign Gold Bond"
allocation = 0.10
"""
    with pytest.raises(ValueError, match=r"\[\[sgb\]\]"):
        load_portfolio_details(_write_toml(tmp_path, toml))
