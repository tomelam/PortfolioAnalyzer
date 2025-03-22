#!/bin/bash

PORT_DIR="port"
BENCHMARK="data/NIFTRI.csv"
OUTPUT="portfolio_metrics.csv"

# CSV header
echo "Portfolio,Annualized Return,Volatility,Sharpe Ratio,Sortino Ratio,Drawdowns,Alpha,Beta" > "$OUTPUT"

for file in "$PORT_DIR"/*.toml; do
    label=$(grep -m1 'label' "$file" | cut -d'"' -f2)

    # Capture the output and extract metric values
    metrics=$(python3 main.py "$file" --benchmark-returns-file "$BENCHMARK" --do-not-plot 2>/dev/null)

    ret=$(echo "$metrics" | awk -F: '/Annualized Return/ {gsub(/ /,"",$2); print $2}')
    vol=$(echo "$metrics" | awk -F: '/Volatility/ {gsub(/ /,"",$2); print $2}')
    sharpe=$(echo "$metrics" | awk -F: '/Sharpe Ratio/ {gsub(/ /,"",$2); print $2}')
    sortino=$(echo "$metrics" | awk -F: '/Sortino Ratio/ {gsub(/ /,"",$2); print $2}')
    drawdowns=$(echo "$metrics" | awk -F: '/Drawdowns/ {gsub(/ /,"",$2); print $2}')
    alpha=$(echo "$metrics" | awk -F: '/Alpha/ {gsub(/ /,"",$2); printf "%.2f", $2 * 100}')
    beta=$(echo "$metrics" | awk -F: '/Beta/ {gsub(/ /,"",$2); print $2}')

    echo "$label,$ret,$vol,$sharpe,$sortino,$drawdowns,$alpha,$beta" >> "$OUTPUT"
done

echo -e "\nSaved full metrics to $OUTPUT"
