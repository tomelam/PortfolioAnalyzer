#!/usr/bin/env bash
#
# Render every portfolio in port/ to a PNG + CSV under outputs/<name>/.
#
# Usage:
#   scripts/render-all.sh                       # render every port/*.toml
#   scripts/render-all.sh port/port-1.toml ...  # render only the named ones
#
# Skips any TOML that fails to load (e.g. legacy schema) and continues.
# Exit code = number of failed portfolios.

set -u

cd "$(dirname "$0")/.."

PA="${PA:-./venv/bin/portfolio-analyzer}"
CONFIG="${CONFIG:-tests/fixtures/golden_master_config.toml}"  # bypass staleness gate by default; override when data is fresh
if [[ ! -x "$PA" ]]; then
    echo "❌ $PA not found. Run:  python3.12 -m venv venv && ./venv/bin/python -m pip install -e \".[dev]\"" >&2
    exit 1
fi

if [[ $# -gt 0 ]]; then
    portfolios=("$@")
else
    portfolios=(port/*.toml)
fi

failures=0
for toml in "${portfolios[@]}"; do
    name=$(basename "$toml" .toml)
    out="outputs/$name"
    mkdir -p "$out"

    echo "── $name ──"
    if "$PA" \
        --config "$CONFIG" \
        --quiet \
        --disable-plot-display \
        --output-snapshot \
        --output-csv \
        --output-dir "$out" \
        "$toml"; then
        echo "  ✓ $out/$name.png + $out/$name.csv"
    else
        echo "  ✗ $name failed (see above)" >&2
        failures=$((failures + 1))
    fi
done

if [[ $failures -gt 0 ]]; then
    echo
    echo "$failures portfolio(s) failed." >&2
    exit "$failures"
fi
echo
echo "All portfolios rendered. PNGs and CSVs are under outputs/."
