"""Console entry point to refresh benchmark/risk-free data (cron-able).

Installed as ``portfolio-analyzer-update`` (see pyproject [project.scripts]).
Refreshes every registered source in one shot and prints a short report.
Intended to run on a schedule (e.g. a daily cron) so the pipeline always
reads fresh local CSVs:

    0 6 * * *  cd /path/to/PortfolioAnalyzer && ./venv/bin/portfolio-analyzer-update

Exit code is 0 when at least one source updated, 1 only when every source
failed — so a single flaky feed doesn't page you, but a total outage does.
"""

from __future__ import annotations

import sys

from loaders.data_update import update_all


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
