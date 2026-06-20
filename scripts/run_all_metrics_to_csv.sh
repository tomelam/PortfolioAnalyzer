#!/usr/bin/env bash
#
# Build one combined CSV summarizing every portfolio's headline metrics.
#
# Usage:
#   scripts/run_all_metrics_to_csv.sh                  # scans examples/port/*.toml → portfolio_metrics.csv
#   scripts/run_all_metrics_to_csv.sh some-dir/        # scans some-dir/*.toml
#   scripts/run_all_metrics_to_csv.sh -o my.csv        # writes to my.csv

set -u

cd "$(dirname "$0")/.."

PORT_DIR="examples/port"
OUTPUT="portfolio_metrics.csv"
PA="${PA:-./venv/bin/portfolio-analyzer}"
CONFIG="${CONFIG:-tests/fixtures/golden_master_config.toml}"  # bypass staleness gate by default

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            head -10 "$0" | sed 's/^# \?//'
            exit 0
            ;;
        -o)
            OUTPUT="$2"; shift 2 ;;
        *)
            PORT_DIR="$1"; shift ;;
    esac
done

echo "Portfolio,CAGR,Volatility,Sharpe,Sortino,Alpha,Beta,Drawdowns,Max Drawdown,Max DD Start,DD Days,Recovery Days" > "$OUTPUT"

for f in "$PORT_DIR"/*.toml; do
    name=$(basename "$f" .toml)
    tmp_dir=$(mktemp -d)
    if "$PA" \
        --config "$CONFIG" \
        --quiet --disable-plot-display --output-csv \
        --output-dir "$tmp_dir" \
        "$f" > /dev/null 2>&1; then
        if [[ -f "$tmp_dir/$name.csv" ]]; then
            cat "$tmp_dir/$name.csv" >> "$OUTPUT"
        fi
    else
        echo "  ✗ $name failed" >&2
    fi
    rm -rf "$tmp_dir"
done

echo "✓ Wrote: $OUTPUT"
