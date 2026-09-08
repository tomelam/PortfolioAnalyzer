"""Nudge: fail when webgrab has cut a release newer than the pin we hold.

`webgrab` is pinned to a tag in pyproject.toml (see CLAUDE.md, "Bumping webgrab").
A pin is the right call -- an unpinned `git+` URL tracks HEAD, so two installs on
different days get different fetching code -- but a pin with nothing watching it is
how a project ends up a year behind without anyone deciding to be.

This test is the watcher, and it is deliberately quiet:

* It does NOT fire when webgrab's HEAD moves. HEAD moves constantly (five times in
  one evening on 2026-09-07) and firing on that would be noise, which trains people
  to ignore the signal.
* It fires only when a NEWER TAG exists -- a deliberate release, which is exactly
  the moment a bump is worth considering.
* It clears itself. Bump the pin and it goes green again. A gate that cannot go
  green is a gate people learn to skip.

Network tier, so it never blocks offline work; it lands at the merge gate, which is
when the full suite runs anyway.

It **fails** rather than skips when github.com cannot be reached. Ignoring HEAD
movement suppresses a false signal; skipping an unreachable remote would suppress a
true one -- "this gate did not run" -- and a skip reads as green in the summary line.
Fail loud: a check that examined nothing must say so.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPO = "https://github.com/tomelam/webgrab.git"

_PIN_RE = re.compile(r"webgrab\s*@\s*git\+[^@\s]+@(?P<ref>[^\"'\s]+)")
_SEMVER_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _pinned_ref() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for spec in data.get("project", {}).get("dependencies", []):
        m = _PIN_RE.search(spec)
        if m:
            return m.group("ref")
    pytest.fail(
        "no pinned webgrab ref found in [project.dependencies]. "
        "test_non_pypi_dependencies_are_pinned_to_a_ref should have caught this."
    )


def _version(tag: str) -> tuple[int, int, int] | None:
    m = _SEMVER_RE.match(tag)
    return tuple(int(g) for g in m.groups()) if m else None


def _remote_tags(repo: str = REPO) -> list[str]:
    """Tags on the public remote, over the anonymous HTTPS URL pip itself uses."""
    proc = subprocess.run(
        ["git", "ls-remote", "--tags", "--refs", repo],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        # FAIL, do not skip. You are in the network tier because you asked to be;
        # an unreachable remote means this gate did not run, and "528 passed, 1
        # skipped" is indistinguishable from green to whoever is merging. Worse,
        # if only github.com is blocked (proxy, DNS, rate limit) the other network
        # tests still pass and the suite reads fully green while the pin question
        # went unanswered -- a check that reports fine for the wrong reason.
        pytest.fail(
            f"could not reach {repo}, so the webgrab pin was NOT checked.\n"
            f"  git said: {proc.stderr.strip()[:200]}\n"
            "This is not a stale pin - it is an un-run check. Restore network "
            "access and re-run; do not merge on the strength of this suite."
        )
    return [line.rsplit("/", 1)[-1] for line in proc.stdout.splitlines() if line.strip()]


@pytest.mark.network
def test_webgrab_pin_is_the_newest_release():
    pinned = _pinned_ref()
    pinned_v = _version(pinned)
    assert pinned_v is not None, (
        f"webgrab is pinned to {pinned!r}, which is not a vX.Y.Z tag. Pin to a "
        f"semver tag so this check can compare releases."
    )

    releases = sorted(
        (v for v in (_version(t) for t in _remote_tags()) if v is not None)
    )
    assert releases, "webgrab has no vX.Y.Z tags on the remote — nothing to compare"

    newest = releases[-1]
    assert newest <= pinned_v, (
        "webgrab has a newer release than the pin.\n"
        f"  pinned : v{'.'.join(map(str, pinned_v))}\n"
        f"  newest : v{'.'.join(map(str, newest))}\n"
        "This is a nudge, not a defect — the current pin still works. Follow the "
        "bump ritual in CLAUDE.md ('Bumping webgrab'): update pyproject, reinstall, "
        "and run the FULL suite including the network tier before committing."
    )
