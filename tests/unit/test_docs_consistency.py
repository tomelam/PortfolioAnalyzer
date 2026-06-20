"""Guard against README / QUICKSTART (and config) drift from reality.

Documented flags, config keys, and data files have repeatedly fallen out of sync
with the code — e.g. a ``--save-output-to`` flag that never existed, a
``benchmark_file`` config key the loader silently ignores, a ``--save-golden-data``
flag that was removed. These tests validate the docs' *claims* against the code's
own sources of truth:

* CLI flags  → ``main.build_parser()`` (the real argparse option set)
* config keys → the ``config.get("…")`` calls in ``main.build_settings``
* data files → the real loaders (``loaders.benchmark`` / ``loaders.risk_free``)

They run fully offline. They intentionally do **not** execute install/network
snippets (that would test pip and niftyindices, not the docs), and they can't
catch prose claims or value judgments — those still need human review.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import main
from loaders.benchmark import load_timeseries_csv
from loaders.risk_free import fetch_and_standardize_risk_free_rates

ROOT = Path(__file__).resolve().parents[2]
DOC_FILES = [ROOT / "README.md", ROOT / "QUICKSTART.md"]
CONFIG_FILES = sorted((ROOT / "config").glob("*.toml"))

# Third-party flags that legitimately appear in install/run snippets and are not
# the analyzer's to define: pip (-e), python/pytest (-m), rm (-rf), make (-j).
_FOREIGN_FLAGS = {"-e", "-m", "-rf", "-j"}

# Removed flags kept in the parser only as fail-fast tombstones; docs must not
# advertise them, and the coverage check must not require them.
_TOMBSTONES = {"--skip-age-check", "--no-auto-update", "--max-riskfree-delay"}


# --- helpers ---------------------------------------------------------------

def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _inline_spans(text: str) -> list[str]:
    """Inline `code` spans only."""
    return re.findall(r"`([^`\n]+)`", text)


_FENCED_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _code_spans(text: str) -> list[str]:
    """Fenced ```code``` blocks + inline `code` spans of a markdown doc."""
    return _FENCED_RE.findall(text) + _inline_spans(text)


def _fenced_command_lines(text: str) -> list[str]:
    """Runnable command lines from fenced blocks (prompt-stripped, no comments)."""
    lines: list[str] = []
    for block in _FENCED_RE.findall(text):
        for raw in block.splitlines():
            line = raw.strip().removeprefix("$").strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


_FLAG_RE = re.compile(r"(?<![\w/])(--?[A-Za-z][\w-]*)")
_KEY_ASSIGN_RE = re.compile(r"^([a-z_][\w]*)\s*=")
_CONFIG_GET_RE = re.compile(r"""config\.get\(\s*["']([a-zA-Z_]\w*)["']""")


def _documented_flags() -> set[str]:
    flags: set[str] = set()
    for doc in DOC_FILES:
        for span in _code_spans(_read(doc)):
            for tok in _FLAG_RE.findall(span):
                if tok.startswith("-") and not tok.startswith("---"):
                    flags.add(tok)
    return flags - _FOREIGN_FLAGS


def _real_option_strings() -> set[str]:
    opts: set[str] = set()
    for action in main.build_parser()._actions:  # noqa: SLF001 - argparse's introspection point
        opts.update(action.option_strings)
    return opts


def _documented_config_keys() -> set[str]:
    # Inline spans only: runtime config keys are documented as inline code
    # (`output_csv = true`), whereas fenced ```toml blocks may show *portfolio*
    # TOML (label/allocation/…), which are not runtime-config keys.
    keys: set[str] = set()
    for doc in DOC_FILES:
        for span in _inline_spans(_read(doc)):
            m = _KEY_ASSIGN_RE.match(span.strip())
            if m:
                keys.add(m.group(1))
    return keys


def _real_config_keys() -> set[str]:
    return set(_CONFIG_GET_RE.findall(_read(ROOT / "main.py")))


def _console_script_names() -> set[str]:
    data = tomllib.loads(_read(ROOT / "pyproject.toml"))
    return set(data.get("project", {}).get("scripts", {}))


# --- invocation form -------------------------------------------------------

def test_docs_invoke_pa_wrapper_not_bare_console_script():
    """Runnable doc examples must use ``./pa`` (or ``./venv/bin/<script>``), never
    the bare console script — which only resolves with an activated venv. ``./pa``
    is the one canonical entry point; this guards against the bare form creeping
    back into the examples (it passed review once before).
    """
    scripts = _console_script_names()
    assert scripts, "no [project.scripts] entries found in pyproject.toml"
    offenders = [
        (doc.name, line)
        for doc in DOC_FILES
        for line in _fenced_command_lines(_read(doc))
        if line.split()[0] in scripts
    ]
    assert not offenders, (
        "Runnable doc examples invoke the bare console script (needs an activated "
        f"venv) instead of the canonical ./pa wrapper: {offenders}. "
        "Use './pa …' or './venv/bin/<script> …'."
    )


# --- CLI flags -------------------------------------------------------------

def test_documented_cli_flags_exist():
    bogus = sorted(_documented_flags() - _real_option_strings())
    assert not bogus, (
        f"README/QUICKSTART reference CLI flags that main.build_parser() does "
        f"not define: {bogus}. Fix the docs (or the parser)."
    )


def test_all_cli_flags_are_documented():
    long_flags = {
        o for o in _real_option_strings()
        if o.startswith("--") and o != "--help" and o not in _TOMBSTONES
    }
    doc_text = "\n".join(_read(d) for d in DOC_FILES)
    undocumented = sorted(f for f in long_flags if f not in doc_text)
    assert not undocumented, (
        f"These real CLI flags are mentioned nowhere in README/QUICKSTART: "
        f"{undocumented}. Document them (or remove if obsolete)."
    )


# --- config keys -----------------------------------------------------------

def test_documented_config_keys_exist():
    bogus = sorted(_documented_config_keys() - _real_config_keys())
    assert not bogus, (
        f"README/QUICKSTART document config keys that build_settings() never "
        f"reads: {bogus} (e.g. benchmark_file vs benchmark_returns_file)."
    )


def test_config_toml_files_use_real_keys():
    real = _real_config_keys()
    offenders = {
        path.name: sorted(k for k in tomllib.loads(_read(path)) if k not in real)
        for path in CONFIG_FILES
    }
    offenders = {name: bad for name, bad in offenders.items() if bad}
    assert not offenders, (
        f"config/*.toml set keys build_settings() never reads: {offenders}."
    )


# --- data files ------------------------------------------------------------

@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.name)
def test_config_data_files_load(config_path):
    """Each config's benchmark / risk-free file loads with its stated date format.

    Catches a stale or wrong filename (missing file) and a wrong date_format
    (parse error) — the semantic doc bugs a flag/key check can't see. Offline:
    reads the committed CSVs via the real loaders with the staleness gate off.
    """
    cfg = tomllib.loads(_read(config_path))
    checked = False

    bench = cfg.get("benchmark_returns_file")
    if bench:
        checked = True
        assert (ROOT / bench).exists(), f"{config_path.name}: missing file {bench!r}"
        ts = load_timeseries_csv(str(ROOT / bench), cfg.get("benchmark_date_format", "%m/%d/%Y"))
        assert len(ts.value_series()) > 0, f"{config_path.name}: {bench!r} parsed empty"

    rf = cfg.get("risk_free_rates_file")
    if rf:
        checked = True
        assert (ROOT / rf).exists(), f"{config_path.name}: missing file {rf!r}"
        series = fetch_and_standardize_risk_free_rates(
            str(ROOT / rf), cfg.get("riskfree_date_format", "%Y-%m-%d"), None
        )
        assert len(series) > 0, f"{config_path.name}: {rf!r} parsed empty"

    if not checked:
        pytest.skip("no benchmark/risk-free files declared in this config")
