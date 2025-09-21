import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from portfolio_methods import *

def demo_time_based_rebalancing():
    # Step 1: Define your portfolio and target weights
    us_tickers = ['VTI', 'VXUS', 'BND']
    weights = [0.60, 0.20, 0.20]  # A sample 60/20/20 allocation

    # Example for time-based rebalancing
    initial_investment_time_based = 10000
    start_date_time_based = '2024-09-01'
    rebalance_date = '2025-09-01'

    # Fetch data up to the rebalance date
    portfolio_data = get_historical_data(us_tickers, start_date=start_date_time_based, end_date=rebalance_date)

    if portfolio_data  is not None and not portfolio_data.empty:
        # Calculate the total value on the rebalance date
        total_value_on_rebalance_date = calculate_portfolio_value_on_date(portfolio_data,
                                                                          initial_investment_time_based,
                                                                          start_date_time_based,
                                                                          weights)

        # Check for rebalancing on the specific date
        if total_value_on_rebalance_date:
            check_rebalance_at_date(portfolio_data, weights, rebalance_date, total_value_on_rebalance_date)


def demo_tolerance_based_rebalancing():
    # Example for tolerance-based rebalancing
    us_tickers = ['VTI', 'VXUS', 'BND']
    weights = [0.60, 0.20, 0.20]  # A sample 60/20/20 allocation
    rebalance_tolerance = 0.05  # 5% tolerance

    start_date = '2024-09-01'

    portfolio_data_tolerance = get_historical_data(us_tickers, start_date=start_date)

    if portfolio_data_tolerance is not None and not portfolio_data_tolerance.empty:
        simple_rebalance_strategy(portfolio_data_tolerance, weights, rebalance_tolerance)



if __name__ == "__main__":
    # Example Time-Based rebalance US portfolio
    # demo_time_based_rebalancing()
    demo_tolerance_based_rebalancing()

