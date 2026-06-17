#!/bin/bash

for f in config/*.toml; do
    clear
    echo "========================================"
    echo "Running with config: $(basename "$f")"
    echo "========================================"

    # Optional: clean outputs directory
    rm -f outputs/*

    ./venv/bin/portfolio-analyzer port/port-hdfc-midcap.toml --config "$f"

    echo ""
    echo "🕵️  Inspect output files (e.g., in outputs/), then press Enter to continue..."
    read
    echo ""
done
