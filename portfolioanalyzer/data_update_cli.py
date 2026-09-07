"""Console entry point for a manual one-shot refresh of benchmark/risk-free data.

Installed as ``portfolio-analyzer-update`` (see pyproject [project.scripts]).
Refreshes every registered source in one shot and prints a short report.

This is **optional** — a normal ``portfolio-analyzer`` run already refreshes
whatever reference feed is behind its cadence (see ``docs/DATA_REFRESH.md``).
It is handy to warm the local CSVs ahead of time, e.g. before an offline
session. There is deliberately no scheduled/cron path.

Arguments are parsed before anything is fetched. Until 2026-09-07 ``main`` took
``argv`` and ignored it, so ``portfolio-analyzer-update --help`` -- the command
someone runs to find out what this does -- refreshed every registered source
instead of printing help. One of those sources holds blocks against the source
IP, so an informational command could cost real access.

Exit code is 0 when at least one source updated, 1 only when every source
failed — so a single flaky feed doesn't fail the command, but a total outage
does.
"""

from __future__ import annotations

import argparse
import sys

from portfolioanalyzer.loaders.data_update import REGISTRY, update_all


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="portfolio-analyzer-update",
        description=(
            "Refresh every registered reference feed in one shot and print a "
            "short report. Optional: a normal portfolio-analyzer run already "
            "refreshes whatever feed is behind its cadence."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be refreshed and exit, without contacting any host",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.dry_run:
        print("Would refresh:")
        for name, src in REGISTRY.items():
            label = getattr(src, "label", "") or name
            print(f"  {name:<24} {label}")
        print(f"\n{len(REGISTRY)} source(s). Nothing was contacted.")
        return 0

    results = update_all()
    succeeded = 0
    for r in results:
        if r.get("ok"):
            succeeded += 1
            print(f"✅ {r['name']}: {r['rows']} rows through {r['last_date']}")
        else:
            print(f"⚠️  {r['name']}: FAILED — {r.get('error')}", file=sys.stderr)
    print(f"\n{succeeded}/{len(results)} source(s) updated.")
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
