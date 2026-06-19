#!/usr/bin/env bash
#
# Run the single-asset sanity portfolios one after another.
# Each pops up a matplotlib plot window and prints metrics to stdout.
# Use --headless / -H to suppress the windows.

set -u

cd "$(dirname "$0")/.."

PA="./venv/bin/portfolio-analyzer"
HEADLESS=""
if [[ "${1:-}" == "-H" || "${1:-}" == "--headless" ]]; then
    HEADLESS="--disable-plot-display"
fi

for p in port-ppf port-scss port-1 port-sgb port-gold; do
    echo
    echo "════ $p ════"
    "$PA" $HEADLESS "port/$p.toml" || echo "  ✗ $p failed" >&2
done
