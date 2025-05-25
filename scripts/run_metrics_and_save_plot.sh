#!/bin/bash
# Usage: ./run_metrics_and_save_plot.sh path/to/portfolio.toml

set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 path/to/portfolio.toml"
  exit 1
fi

TOML_FILE="$1"
PORTFOLIO_NAME=$(basename "$TOML_FILE" .toml)
OUTPUT_DIR="outputs/$PORTFOLIO_NAME"

mkdir -p "$OUTPUT_DIR"

python3 main.py "$TOML_FILE" \
  --disable-plot-display \
  --output-dir "$OUTPUT_DIR" \
  --output-csv \
  --output-snapshot
