def plot_cumulative_returns(
    portfolio_label,
    cumulative_historical,
    title,
    benchmark_cumulative=None,
    benchmark_name=None,
    allocations=None,
    metrics=None,
    max_drawdowns=None
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import pandas as pd

    # If either series is empty, just plot whatever we have.
    if cumulative_historical.empty and (benchmark_cumulative is None or benchmark_cumulative.empty):
        print("Nothing to plot: both portfolio and benchmark are empty.")
        return

    # Find the earliest date in each series:
    port_start = cumulative_historical.index[0] if not cumulative_historical.empty else None
    bench_start = benchmark_cumulative.index[0] if (benchmark_cumulative is not None and not benchmark_cumulative.empty) else None

    # Choose the rebase date as the maximum (later) of the two starts.
    if port_start is not None and bench_start is not None:
        rebase_date = max(port_start, bench_start)
    elif port_start is not None:
        rebase_date = port_start
    else:
        rebase_date = bench_start
    print("DEBUG: rebase_date =", rebase_date)
    print("DEBUG: cumulative_historical starts at", cumulative_historical.index[0])
    print("DEBUG: cumulative_historical ends at", cumulative_historical.index[-1])

    def rebase_series(s):
        if rebase_date < s.index[0]:
            # If rebase_date is still earlier than the series starts, use s.index[0].
            base_date = s.index[0]
        else:
            # Or use asof() in case rebase_date is in the gap:
            base_date = rebase_date
        base_value = s.asof(base_date)
        print("DEBUG: base_date used =", base_date, "base_value =", base_value)
        return s.div(base_value).mul(100)

    # Rebase the portfolio
    print("DEBUG: cumulative_historical head:")
    print(cumulative_historical.head())
    print("DEBUG: cumulative_historical tail:")
    print(cumulative_historical.tail())
    cumulative_historical_rebased = rebase_series(cumulative_historical)
    print("DEBUG: cumulative_historical_rebased head:")
    print(cumulative_historical_rebased.head())
    # Rebase the benchmark if it exists
    benchmark_cumulative_rebased = None
    if benchmark_cumulative is not None and not benchmark_cumulative.empty:
        benchmark_cumulative_rebased = rebase_series(benchmark_cumulative)

    # Now plot
    plt.figure(figsize=(12, 6))
    if not cumulative_historical_rebased.empty:
        plt.plot(cumulative_historical_rebased, label="Historical Portfolio Relative Value", color="blue")
    if benchmark_cumulative_rebased is not None and not benchmark_cumulative_rebased.empty:
        plt.plot(benchmark_cumulative_rebased, label=benchmark_name, color="green")

    # Mark the rebase date (dotted vertical line)
    plt.axvline(rebase_date, color="gray", linestyle=":", label="Rebase Date")

    # Legend with drawdowns patch
    red_patch = mpatches.Patch(color='red', alpha=0.1, label="Significant Drawdowns")
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(red_patch)
    labels.append("Significant Drawdowns")
    plt.legend(handles=handles, labels=labels, loc='upper left')

    # Show the allocations box
    if allocations is not None:
        allocation_text = "\n".join(
            f"{k}: {v*100:.2f}%" for k, v in allocations.items()
        )
        plt.gca().text(
            0.02, 0.75, allocation_text,
            fontsize=9, transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.8)
        )

    # Show the metrics box
    if metrics:
        metrics_text = "\n".join([
            f"{key}: {value * 100:.2f}%" if key in ["Annualized Return", "Volatility"] else
            (f"{key}: {value:d}" if key == "Drawdowns" else
             (f"{key}: {value:.4f}" if isinstance(value, (int, float)) else f"{key}: {value}"))
            for key, value in metrics.items()
        ])
        plt.gca().text(
            0.98, 0.75, metrics_text,
            fontsize=9, transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.8)
        )

    # Highlight drawdown periods
    if max_drawdowns:
        for dd in max_drawdowns:
            plt.axvspan(dd["start_date"], dd["end_date"], color="red", alpha=0.1)

    plt.title(f"{title}: {portfolio_label}")
    plt.xlabel("Date")
    plt.ylabel("Relative Value")
    plt.grid()
    plt.show()
    plt.close()
