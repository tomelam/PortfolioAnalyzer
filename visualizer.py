import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from datetime import datetime

def plot_cumulative_returns(
        portfolio_label,
        cumulative_historical,
        title,
        benchmark_cumulative,
        benchmark_name=None,
        allocations=None,
        metrics=None,
        max_drawdowns=None,
        rebase_date=datetime(2008, 1, 1)
):
    # Compute the start dates for each series:
    portfolio_start = cumulative_historical.index.min()
    benchmark_start = benchmark_cumulative.index.min()
    # The proper rebase date is the later of the two:
    rebase_date = max(portfolio_start, benchmark_start)
    
    # Find the nearest dates in each series if rebase_date is not exactly present:
    if rebase_date not in cumulative_historical.index:
        rebase_date_portfolio = cumulative_historical.index[cumulative_historical.index.get_loc(rebase_date, method='ffill')]
    else:
        rebase_date_portfolio = rebase_date

    if rebase_date not in benchmark_cumulative.index:
        rebase_date_benchmark = benchmark_cumulative.index[benchmark_cumulative.index.get_loc(rebase_date, method='ffill')]
    else:
        rebase_date_benchmark = rebase_date

    # Rebase the portfolio and benchmark series so they equal 100 at the proper rebase date.
    rebased_historical = cumulative_historical / cumulative_historical.loc[rebase_date_portfolio] * 100
    rebased_benchmark = benchmark_cumulative / benchmark_cumulative.loc[rebase_date_benchmark] * 100

    plt.figure(figsize=(12, 6))

    # Plot the rebased portfolio returns
    plt.plot(
        rebased_historical.index,
        rebased_historical,
        label="Historical Portfolio Returns",
        color="blue"
    )

    # Plot the rebased benchmark returns, if available
    if rebased_benchmark is not None and not rebased_benchmark.empty:
        plt.plot(rebased_benchmark.index, rebased_benchmark, label=benchmark_name, color="green")

    # Draw a vertical line at the rebase date (using the portfolio's rebase date)
    plt.axvline(rebase_date_portfolio, color="gray", linestyle=":", label="Rebase Date")

    # Create the legend and add a patch for significant drawdowns
    red_patch = mpatches.Patch(color='red', alpha=0.1, label="Significant Drawdowns")
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(red_patch)
    labels.append("Significant Drawdowns")
    plt.legend(handles=handles, labels=labels, loc='upper left')

    # Place the allocation box slightly higher within the left margin
    if allocations is not None:
        allocations_text = "\n".join([f"{key}: {value * 100:.2f}%" for key, value in allocations.items()])
        plt.gca().text(
            0.02, 0.77, allocations_text,
            fontsize=9, transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.8)
        )

    # Display metrics in the lower-right corner
    if metrics:
        metrics_text = "\n".join([
            f"{key}: {value * 100:.2f}%" if key in ["Annualized Return", "Volatility"] else
            (f"{key}: {value:d}" if key == "Drawdowns" else
             (f"{key}: {value:.4f}" if isinstance(value, (int, float)) else f"{key}: {value}"))
            for key, value in metrics.items()
        ])
        plt.gca().text(
            0.98, 0.26, metrics_text,
            fontsize=9, transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.8)
        )

    # Highlight drawdown periods
    if max_drawdowns:
        for drawdown in max_drawdowns:
            plt.axvspan(drawdown['start_date'], drawdown['end_date'], color='red', alpha=0.1)

    plt.title(f"{title}: {portfolio_label}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.grid()

    # Optional: define a key-press function to toggle zoom
    def toggle_zoom(event):
        ax = plt.gca()
        if event.key == 'z':
            ax.set_xlim(pd.Timestamp('2020-01-01'), ax.get_xlim()[1])
        elif event.key == 'r':
            ax.set_xlim(rebased_historical.index[0], rebased_historical.index[-1])
        plt.draw()

    plt.connect('key_press_event', toggle_zoom)
    plt.show()
    plt.close()
