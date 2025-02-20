import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

def plot_cumulative_returns(portfolio_label, cumulative_historical, title, benchmark_cumulative=None, benchmark_name=None, allocations=None, metrics=None, max_drawdowns=None):
    plt.figure(figsize=(12, 6))

    # Plot historical portfolio returns
    plt.plot(cumulative_historical * 100, label="Historical Portfolio Returns", color="blue")

    # Optionally plot benchmark returns
    if benchmark_cumulative is not None:
        plt.plot(benchmark_cumulative * 100, label=benchmark_name, color="green")

    # Add vertical line for portfolio start
    plt.axvline(cumulative_historical.index[0], color="gray", linestyle=":", label="Portfolio Start")

    # Create the legend first
    red_patch = mpatches.Patch(color='red', alpha=0.1, label="Significant Drawdowns")
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(red_patch)
    labels.append("Significant Drawdowns")
    plt.legend(handles=handles, labels=labels, loc='upper left')

    # Place the allocation box slightly higher within the left margin, without shifting right
    if allocations:
        allocations_text = "\n".join([f"{key}: {value * 100:.2f}%" for key, value in allocations.items()])
        plt.gca().text(
            0.02, 0.77, allocations_text,  # x remains near left; y raised slightly
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

    def toggle_zoom(event):
        ax = plt.gca()
        if event.key == 'z':
            ax.set_xlim(pd.Timestamp('2020-01-01'), ax.get_xlim()[1])
        elif event.key == 'r':
            ax.set_xlim(cumulative_historical.index[0], cumulative_historical.index[-1])
        plt.draw()

    plt.connect('key_press_event', toggle_zoom)
    plt.show()
    plt.close()
