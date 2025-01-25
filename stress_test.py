import pandas as pd
import numpy as np
from datetime import timedelta


# Simulate and visualize multiple shocks
def simulate_multiple_shocks(
    portfolio_label,
    latest_portfolio_value,
    annualized_return,
    portfolio_allocations,
    shock_scenarios,
):
    """
    Simulate multiple shock scenarios based on the annualized return and portfolio asset allocations.

    Parameters:
        latest_portfolio_value (float): The last portfolio value.
        annualized_return (float): The historical annualized return.
        portfolio_allocations (dict): Asset allocations as percentages (e.g., {'equity': 0.6, 'debt': 0.3, 'cash': 0.1}).
        shock_scenarios (list of dict): Each dict contains shocks for equity, debt, and cash.

    Returns:
        dict: Keys are scenario names, and values are tuples (plot_color, projected_portfolio_values).
    """
    scenario_results = {}
    daily_growth_rate = (1 + annualized_return) ** (
        1 / 252
    ) - 1  # Daily growth rate from annualized return

    for scenario in shock_scenarios:
        name = scenario["name"]
        equity_shock = scenario["equity_shock"]
        debt_shock = scenario["debt_shock"]
        # cash_shock = scenario["cash_shock"]
        shock_day = scenario["shock_day"]
        projection_days = scenario["projection_days"]
        plot_color = scenario["plot_color"]

        # Initialize portfolio values
        portfolio_values = [latest_portfolio_value]

        # Simulate daily growth for the projection period
        for day in range(projection_days):
            if day == shock_day:
                # Apply shocks based on portfolio allocations
                shock_factor = (
                    1
                    + (equity_shock * portfolio_allocations["equity"])
                    + (debt_shock * portfolio_allocations["debt"])
                )
                new_value = portfolio_values[-1] * shock_factor
            else:
                # Simulate daily growth (example: 0.01% daily growth)
                daily_growth_rate = 0.0005  # Adjust as needed
                new_value = portfolio_values[-1] * (1 + daily_growth_rate)

            portfolio_values.append(new_value)

        # Convert to a Pandas Series for easy plotting
        projected_portfolio_values = pd.Series(
            portfolio_values
        )  # Generate projected returns

        # Store the scenario results
        scenario_results[name] = (plot_color, projected_portfolio_values)

    return scenario_results
