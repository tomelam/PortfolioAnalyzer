#!/usr/bin/env bash
#
# Render a PNG plot for every port/*.toml into the chosen output directory.
# (CSVs are not produced; use scripts/render-all.sh for PNG + CSV per
# portfolio under outputs/<name>/, or scripts/run_all_metrics_to_csv.sh
# for a single combined CSV.)

set -u

cd "$(dirname "$0")/.."

PORT_DIR="port"
OUT_DIR="plots"
PA="${PA:-./venv/bin/portfolio-analyzer}"
CONFIG="${CONFIG:-tests/fixtures/golden_master_config.toml}"  # bypass staleness gate by default

mkdir -p "$OUT_DIR"

for file in "$PORT_DIR"/*.toml; do
    name=$(basename "$file" .toml)
    echo "Plotting $name..."
    "$PA" \
        --config "$CONFIG" \
        --quiet \
        --disable-plot-display \
        --output-snapshot \
        --output-dir "$OUT_DIR/$name" \
        "$file" || echo "  ✗ $name failed" >&2
done

echo "Saved plots under $OUT_DIR/<portfolio>/"
