#!/usr/bin/env bash
# Usage: scripts/run_metrics_and_save_plot.sh path/to/portfolio.toml
#
# Render a single portfolio to outputs/<name>/{<name>.png, <name>.csv}.

set -eu

cd "$(dirname "$0")/.."

if [ $# -ne 1 ]; then
  echo "Usage: $0 path/to/portfolio.toml"
  exit 1
fi

TOML_FILE="$1"
PORTFOLIO_NAME=$(basename "$TOML_FILE" .toml)
OUTPUT_DIR="outputs/$PORTFOLIO_NAME"

mkdir -p "$OUTPUT_DIR"

PYTHON="${PYTHON:-./venv/bin/python}"
CONFIG="${CONFIG:-tests/fixtures/golden_master_config.toml}"

"$PYTHON" main.py \
    --config "$CONFIG" \
    --quiet \
    --disable-plot-display \
    --output-snapshot \
    --output-csv \
    --output-dir "$OUTPUT_DIR" \
    "$TOML_FILE"
