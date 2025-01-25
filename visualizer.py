import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.getLogger("matplotlib").setLevel(logging.WARNING)
import pandas as pd


def plot_cumulative_returns(
    portfolio_label,
    cumulative_historical,
    cumulative_projected,
    title,
    benchmark_cumulative=None,
    benchmark_name=None,
    allocations=None,
    metrics=None,
    max_drawdowns=None,
):
    """
    Plot cumulative historical and projected returns with optional benchmark.

    Parameters:
        cumulative_historical (pd.Series): Cumulative historical returns.
        cumulative_projected (pd.Series): Cumulative projected returns.
        benchmark_cumulative (pd.Series, optional): Cumulative benchmark returns.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(12, 6))

    # Adjust cumulative_projected to start at the end of cumulative_historical
    adjusted_projected = cumulative_projected.copy()
    adjusted_projected.index = cumulative_historical.index[-1] + pd.to_timedelta(
        adjusted_projected.index, unit="D"
    )

    # Plot historical and projected returns
    plt.plot(
        cumulative_historical * 100, label="Historical Portfolio Returns", color="blue"
    )
    plt.plot(
        adjusted_projected * 100,
        label="Projected Portfolio Returns",
        color="red",
        linestyle="--",
    )

    # Optionally plot benchmark returns
    if benchmark_cumulative is not None:
        plt.plot(benchmark_cumulative * 100, label=benchmark_name, color="green")

    # Add a vertical line to denote portfolio start
    plt.axvline(
        cumulative_historical.index[0],
        color="gray",
        linestyle=":",
        label="Portfolio Start",
    )

    # Add a vertical line to denote projection start
    plt.axvline(
        cumulative_historical.index[-1],
        color="gray",
        linestyle=":",
        label="Projection Start",
    )

    # Display allocations just under the legend
    if allocations:
        allocations_text = "\n".join(
            [f"{key}: {value * 100:.2f}%" for key, value in allocations.items()]
        )
        plt.gca().text(
            0.02,
            0.68,
            allocations_text,  # Slightly below the legend (adjust `y` as needed)
            fontsize=9,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.8
            ),
        )

    # Display metrics in small letters in the lower-right corner
    if metrics:
        metrics_text = "\n".join(
            [
                (
                    f"{key}: {value * 100:.2f}%"
                    if key in ["Annualized Return", "Volatility"]
                    else (
                        f"{key}: {value:d}"
                        if key == "Drawdowns"
                        else (
                            f"{key}: {value:.4f}"
                            if isinstance(value, (int, float))
                            else f"{key}: {value}"
                        )
                    )
                )
                for key, value in metrics.items()
            ]
        )
        plt.gca().text(
            0.98,
            0.26,
            metrics_text,
            fontsize=9,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.8
            ),
        )

    # Store the original x-axis limits
    original_xlim = plt.gca().get_xlim()

    plt.title(portfolio_label + " under " + title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.legend(loc="upper left")
    plt.grid()

    # Add toggle for zooming
    def toggle_zoom(event):
        ax = plt.gca()  # Get the current axes
        if event.key == "z":  # Zoom in to recent years
            ax.set_xlim(pd.Timestamp("2020-01-01"), plt.gca().get_xlim()[1])
        elif event.key == "r":  # Reset to full view
            start_date = cumulative_historical.index[0]
            ax.set_xlim(original_xlim)
            # Re-add the vertical gray dotted line
            ax.axvline(start_date, color="gray", linestyle=":", label="Portfolio Start")
        plt.draw()

    if max_drawdowns:
        for drawdown in max_drawdowns:
            plt.axvspan(
                drawdown["start_date"], drawdown["end_date"], color="red", alpha=0.1
            )

    red_patch = mpatches.Patch(color="red", alpha=0.1, label="Significant Drawdowns")
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(red_patch)
    labels.append("Significant Drawdowns")
    plt.legend(
        handles=handles, labels=labels, loc="upper left"
    )  # Adjust `loc` as needed

    plt.connect("key_press_event", toggle_zoom)
    plt.show()
    plt.close()
