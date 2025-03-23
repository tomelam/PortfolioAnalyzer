import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from datetime import datetime
import os
import toml


def display_toml_below_figure(ax_table, toml_file):
    """
    Reads a TOML file, extracts relevant portfolio data, and displays it as a formatted table inside a subplot.
    
    Args:
        ax_table (matplotlib.axes.Axes): The subplot where the table is added.
        toml_file (str): Path to the TOML file.
    """
    # Ensure file exists
    if not os.path.isfile(toml_file):
        raise FileNotFoundError(f"File not found: {toml_file}")

    # Load TOML file
    with open(toml_file, "r") as f:
        data = toml.load(f)

    # Extract label
    file_name = os.path.basename(toml_file)
    label = data.get("label", "Unknown Portfolio")

    # Extract relevant details into table format
    table_data = []
    for key, value in data.items():
        if key in ["file", "label"]:
            continue

        if isinstance(value, dict) and "allocation" in value:
            table_data.append([
                key.capitalize(),
                value["name"].strip(),
                f"{value['allocation'] * 100:6.2f}%"
            ])
        elif isinstance(value, list):
            for fund in value:
                table_data.append([
                    "Fund",
                    fund["name"].strip(),
                    f"{fund['allocation'] * 100:6.2f}%"
                ])

    # Desired columns: Type (1.25x), Asset (4.0x), Allocation (1.25x)
    # But colWidths must be FRACTIONAL, so let's do ratio = 1.25 : 4.0 : 1.25 => total 6.5
    # Convert to fraction:
    type_ratio = 0.8 / 4.5
    asset_ratio = 3.0 / 4.5
    allocation_ratio = 0.7 / 4.5

    col_labels = ["Type", "Asset", "Allocation"]
    ax_table.axis("off")

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='left',
        loc='upper left',
        colWidths=[type_ratio, asset_ratio, allocation_ratio],  # <-- Column width ratios
        bbox=[0.05, 0, 0.65, 1]
    )

    # Optional: reduce row height, fix font size
    table.scale(1, 0.5)
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Format the table
    for (row, col), cell in table._cells.items():
        # Header row
        if row == 0:
            cell.set_text_props(weight='bold', ha='left', fontname='sans-serif')
            cell.set_facecolor("#f0f0f0")

        # Left-align "Type" & "Asset" columns 
        if col == 1:
            cell.PAD = 0.035   # Remove extra internal padding

        # Right-align "Allocation" column
        if col == 2:
            text_obj = cell.get_text()
            text_obj.set_ha("right")

        # Soften border lines
        cell.set_edgecolor("lightgray")
    

def toggle_zoom(event):
    ax = event.canvas.figure.axes[0]  # or whichever axes you want
    if event.key == 'z':
        # Example zoom: set x-limits from 2020-01-01 to current max
        ax.set_xlim(pd.Timestamp('2020-01-01'), ax.get_xlim()[1])
    elif event.key == 'r':
        # 'r' to reset the full range
        ax.relim()      # recalc limits
        ax.autoscale()  # reset to full data
    event.canvas.draw_idle()


def plot_cumulative_returns(
        portfolio_label,
        cumulative_historical,
        title,
        toml_file,
        benchmark_cumulative,
        benchmark_name=None,
        allocations=None,
        metrics=None,
        max_drawdowns=None,
        rebase_date=datetime(2008, 1, 1)
):
    fig = plt.figure(figsize=(10, 6), constrained_layout=True, dpi=200)  # higher DPI for crisp screen rendering

    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.2])
    ax = fig.add_subplot(gs[0])

    # Compute the start dates for each series:
    portfolio_start = cumulative_historical.index.min()
    benchmark_start = benchmark_cumulative.index.min()
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

    # Plot the rebased portfolio returns
    ax.plot(
        rebased_historical.index,
        rebased_historical,
        label="Historical Portfolio Returns",
        color="blue"
    )

    # Plot the rebased benchmark returns, if available
    if rebased_benchmark is not None and not rebased_benchmark.empty:
        ax.plot(rebased_benchmark.index, rebased_benchmark, label=benchmark_name, color="green")

    # Draw a vertical line at the rebase date (using the portfolio's rebase date)
    ax.axvline(rebase_date_portfolio, color="gray", linestyle=":", label="Rebase Date")

    # Highlight drawdown periods
    if max_drawdowns:
        for drawdown in max_drawdowns:
            ax.axvspan(drawdown['start_date'], drawdown['recovery_date'], color='red', alpha=0.1)
    
    # Add allocations and metrics inside the figure
    if allocations is not None:
        #allocations_text = "\n".join([f"{key}: {value * 100:.2f}%" for key, value in allocations.items()])
        #allocations_text = "\n".join([
        #    f"{asset + ':':<18}{weight * 100:>6.2f}%"
        #    for asset, weight in allocations.items()
        #])
        allocations_text = "\n".join([
            f"{asset.replace('_', ' ') + ':':<12}{weight * 100:>6.2f}%"
            for asset, weight in allocations.items()
        ])
        ax.text(0.02, 0.38, allocations_text, fontfamily='monospace', fontsize=9,
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.8))

    if metrics:
        metrics_text = "\n".join([
            f"{key + ':':<18}{value * 100:>6.2f}%" if key in ["Annualized Return", "Volatility", "Alpha"]
            else f"{key + ':':<18}{int(value):>3}" if key == "Drawdowns" and isinstance(value, (int, float))
            else f"{key + ':':<18}{value:>8.4f}" if isinstance(value, (int, float))
            else f"{key + ':':<18}{value}"
            for key, value in metrics.items()
        ])
        ax.text(0.02, 0.70, metrics_text, fontfamily='monospace', fontsize=9, transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.8))
    
    # Set labels and title
    ax.set_title(f"{title}: {portfolio_label}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of Initial Investment (Base = 100)")
    ax.grid()

    # Create the legend
    red_patch = mpatches.Patch(color='red', alpha=0.1, label="Significant Drawdowns")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(red_patch)
    labels.append("Significant Drawdowns")
    ax.legend(handles=handles, labels=labels, loc='upper left')

    # Add asset allocations table in the lower portion
    ax_table = fig.add_subplot(gs[1])
    display_toml_below_figure(ax_table, toml_file)

    # Show plot
    #plt.subplots_adjust(hspace=0.4)  # bigger gap between subplots
    #plt.tight_layout()
    # plt.savefig("portfolio_performance.png", dpi=600, bbox_inches="tight")  # For high-res output
    fig.canvas.mpl_connect('key_press_event', toggle_zoom)
    plt.show()
    plt.close()
