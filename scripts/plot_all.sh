#!/bin/bash

PORT_DIR="port"
OUT_DIR="plots"
BENCHMARK="data/NIFTRI.csv"

mkdir -p "$OUT_DIR"

for file in "$PORT_DIR"/*.toml; do
    echo "Plotting $file..."
    python3 main.py "$file" --benchmark-returns-file "$BENCHMARK" --output-dir "$OUT_DIR"
done

echo "Saved plots to $OUT_DIR/"
