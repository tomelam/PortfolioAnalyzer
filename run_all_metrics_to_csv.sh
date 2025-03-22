#!/bin/bash

# Default values
PORT_DIR="port"
OUTPUT="portfolio_metrics.csv"
BENCHMARK="data/NIFTRI.csv"

# Usage message
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [PORT_DIR]"
    echo "Scans all .toml files in PORT_DIR (default: 'port') and generates a CSV summary."
    exit 0
fi

# Allow custom portfolio directory as optional argument
if [[ -n "$1" ]]; then
    PORT_DIR="$1"
fi

# CSV header
echo "Portfolio,CAGR,Volatility,Sharpe,Sortino,Drawdowns,Max Drawdown,Max DD Start,DD Days,Recovery Days,Alpha,Beta" > "$OUTPUT"

# Loop through portfolios
for file in "$PORT_DIR"/*.toml; do
    python3 main.py "$file" --benchmark-returns-file "$BENCHMARK" --do-not-plot --csv-output >> "$OUTPUT"
done

echo "Saved: $OUTPUT"
