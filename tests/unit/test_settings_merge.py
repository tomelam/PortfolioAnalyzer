"""Unit tests for ``main.build_settings`` — the CLI/config/default merge.

Covers the "TOML-based settings", "config-file settings" and "CLI override
handling" sections of ``tests/TODO.md``. ``build_settings`` is the pure seam
where precedence (CLI > config-file > built-in default) is decided, so it can
be exercised directly without argparse, the network, or ``main()``.

The ``args`` namespace built by :func:`_args` mirrors the real argparse
defaults in ``main.parse_arguments`` (absent optional flags arrive as ``None``
or ``False``), so these tests track the actual CLI surface.
"""

from __future__ import annotations

import types

import pandas as pd
import pytest

import main
from data_loader import load_config_toml

# Built-in defaults build_settings falls back to when neither CLI nor config
# supplies a value — pinned here so a silent change to a default is caught.
DEFAULT_BENCHMARK_FILE = "data/NIFTY Total Returns Historical Data.csv"
DEFAULT_RISK_FREE_FILE = "data/INDIRLTLT01STM.csv"


def _args(**overrides):
    """An argparse-like namespace with the same defaults as a bare CLI run."""
    base = dict(
        toml_file="port/port-1.toml",
        show_plot=None,  # set_defaults(show_plot=None); --disable-plot-display → False
        output_snapshot=False,
        output_csv=False,
        output_dir=None,
        max_drawdown_threshold=None,  # no argparse default → config/built-in owns it
        metrics_method=None,  # ditto
        allow_stale=False,
        quiet=False,
        debug=False,
        lookback=None,
        as_of=None,
        replay_from=None,
        save_replay=None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


# --- built-in defaults -----------------------------------------------------

def test_defaults_when_neither_cli_nor_config_set() -> None:
    s = main.build_settings(_args(), {})
    assert s["portfolio_file"] == "port/port-1.toml"
    assert s["show_plot"] is True
    assert s["output_snapshot"] is False
    assert s["output_csv"] is False
    assert s["output_dir"] == "outputs"
    assert s["drawdown_threshold"] == 5.0
    assert s["metrics_method"] == "daily"
    assert s["allow_stale"] is False
    assert s["quiet"] is False
    assert s["debug"] is False
    assert s["lookback"] is None
    assert s["use_benchmark"] is True
    assert s["benchmark_name"] == "NIFTY Total Returns Index"
    assert s["benchmark_file"] == DEFAULT_BENCHMARK_FILE
    assert s["risk_free_rates_file"] == DEFAULT_RISK_FREE_FILE
    assert s["benchmark_date_format"] == "%m/%d/%Y"
    assert s["riskfree_date_format"] == "%Y-%m-%d"
    assert s["as_of"] is None
    assert s["replay_from"] is None
    assert s["save_replay"] is None


# --- config supplies values when the CLI is silent -------------------------

def test_config_supplies_values_when_cli_absent() -> None:
    config = {
        "output_dir": "cfg_out",
        "output_csv": True,
        "output_snapshot": True,
        "metrics_method": "monthly",
        "max_drawdown_threshold": 12.5,
        "allow_stale": True,
        "quiet": True,
        "lookback": "3Y",
        "show_plot": False,
    }
    s = main.build_settings(_args(), config)
    assert s["output_dir"] == "cfg_out"
    assert s["output_csv"] is True
    assert s["output_snapshot"] is True
    assert s["metrics_method"] == "monthly"
    assert s["drawdown_threshold"] == 12.5
    assert s["allow_stale"] is True
    assert s["quiet"] is True
    assert s["lookback"] == "3Y"
    assert s["show_plot"] is False


def test_config_drawdown_and_metrics_method_are_honored() -> None:
    """Regression: a non-None argparse default for --max-drawdown-threshold /
    --metrics-method used to shadow the config value entirely (the CLI default
    was truthy, so ``args.x or config.get(...)`` never reached the config).
    With the defaults moved into build_settings, config now takes effect."""
    s = main.build_settings(
        _args(), {"max_drawdown_threshold": 10.0, "metrics_method": "monthly"}
    )
    assert s["drawdown_threshold"] == 10.0
    assert s["metrics_method"] == "monthly"


# --- CLI overrides config --------------------------------------------------

def test_cli_overrides_config() -> None:
    config = {
        "output_dir": "cfg_out",
        "output_csv": False,
        "metrics_method": "monthly",
        "max_drawdown_threshold": 12.5,
        "allow_stale": False,
        "quiet": False,
        "lookback": "3Y",
        "show_plot": True,
    }
    args = _args(
        output_dir="cli_out",
        output_csv=True,
        metrics_method="daily",
        max_drawdown_threshold=2.0,
        allow_stale=True,
        quiet=True,
        lookback="1Y",
        show_plot=False,  # --disable-plot-display
    )
    s = main.build_settings(args, config)
    assert s["output_dir"] == "cli_out"
    assert s["output_csv"] is True
    assert s["metrics_method"] == "daily"
    assert s["drawdown_threshold"] == 2.0
    assert s["allow_stale"] is True
    assert s["quiet"] is True
    assert s["lookback"] == "1Y"
    assert s["show_plot"] is False


# --- benchmark / risk-free overrides via config ----------------------------

def test_benchmark_and_risk_free_overrides_via_config() -> None:
    config = {
        "benchmark_returns_file": "data/custom_bench.csv",
        "benchmark_name": "Custom Index",
        "benchmark_date_format": "%d-%m-%Y",
        "riskfree_date_format": "%d/%m/%Y",
        "risk_free_rates_file": "data/custom_rf.csv",
        "use_benchmark": False,
    }
    s = main.build_settings(_args(), config)
    assert s["benchmark_file"] == "data/custom_bench.csv"
    assert s["benchmark_name"] == "Custom Index"
    assert s["benchmark_date_format"] == "%d-%m-%Y"
    assert s["riskfree_date_format"] == "%d/%m/%Y"
    assert s["risk_free_rates_file"] == "data/custom_rf.csv"
    assert s["use_benchmark"] is False


# --- --as-of parsing -------------------------------------------------------

def test_as_of_string_parsed_to_normalized_timestamp() -> None:
    s = main.build_settings(_args(as_of="2026-06-13"), {})
    assert s["as_of"] == pd.Timestamp("2026-06-13")
    assert s["as_of"] == s["as_of"].normalize()  # midnight, no intraday component


def test_as_of_absent_is_none() -> None:
    assert main.build_settings(_args(), {})["as_of"] is None


# --- robustness: unknown keys, missing config ------------------------------

def test_unknown_config_keys_are_ignored() -> None:
    """A config with typos / unrecognized keys must not crash or leak in; the
    merge only consults known keys, so extras are silently dropped."""
    s = main.build_settings(_args(), {"totally_bogus_key": 1, "outpt_dir": "typo"})
    assert "totally_bogus_key" not in s
    assert "outpt_dir" not in s
    assert s["output_dir"] == "outputs"  # the typo'd key did not take effect


def test_missing_config_file_yields_empty_dict_then_defaults(tmp_path) -> None:
    """'CLI works if config is missing': load_config_toml returns {} for a
    non-existent path, and build_settings then produces a complete default
    settings dict."""
    config = load_config_toml(str(tmp_path / "does_not_exist.toml"))
    assert config == {}
    s = main.build_settings(_args(), config)
    assert s["output_dir"] == "outputs"
    assert s["metrics_method"] == "daily"


def test_custom_config_file_is_read(tmp_path) -> None:
    """'Use of a custom config file': load_config_toml reads the given path and
    its values flow through build_settings."""
    cfg = tmp_path / "other.toml"
    cfg.write_text('output_dir = "from_custom_cfg"\nmetrics_method = "monthly"\n')
    config = load_config_toml(str(cfg))
    s = main.build_settings(_args(), config)
    assert s["output_dir"] == "from_custom_cfg"
    assert s["metrics_method"] == "monthly"


@pytest.mark.parametrize("bad", ["garbage", "5Z", "2026-13-40"])
def test_as_of_invalid_string_raises(bad: str) -> None:
    """An unparseable --as-of must fail loudly (cli() catches the ValueError
    and exits non-zero) rather than silently producing a wrong evaluation
    date. pandas raises DateParseError, a ValueError subclass."""
    with pytest.raises(ValueError):
        main.build_settings(_args(as_of=bad), {})
