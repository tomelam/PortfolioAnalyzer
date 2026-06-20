"""Console entry point for a manual one-shot refresh of benchmark/risk-free data.

Installed as ``portfolio-analyzer-update`` (see pyproject [project.scripts]).
Refreshes every registered source in one shot and prints a short report.

This is **optional** — a normal ``portfolio-analyzer`` run already refreshes
whatever reference feed is behind its cadence (see ``docs/DATA_REFRESH.md``).
It is handy to warm the local CSVs ahead of time, e.g. before an offline
session. There is deliberately no scheduled/cron path.

Exit code is 0 when at least one source updated, 1 only when every source
failed — so a single flaky feed doesn't fail the command, but a total outage
does.
"""

from __future__ import annotations

import sys

from portfolioanalyzer.loaders.data_update import update_all


def main(argv: list[str] | None = None) -> int:
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
