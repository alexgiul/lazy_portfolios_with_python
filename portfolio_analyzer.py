import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from portfolio_methods import get_historical_data, calculate_portfolio_returns, calculate_portfolio_value, \
    calculate_portfolio_value_on_date, check_rebalance_at_date, simple_rebalance_strategy, calculate_volatility, \
    calculate_sharpe_ratio, thousands_formatter, calculate_drawdown, monte_carlo_simulation, calculate_correlation


def demo1():
    # Define your portfolio and a specific investment amount
    tickers = ['VTI', 'VXUS', 'BND']
    weights = [0.60, 0.20, 0.20]  # A sample 60/20/20 allocation
    initial_investment = 10000

    # Set the start date for your investment
    start_date = '2024-01-01'

    # Fetch the data
    portfolio_data = get_historical_data(tickers, start_date=start_date)

    # Calculate daily returns
    daily_returns, _ = calculate_portfolio_returns(portfolio_data, weights)

    # Calculate the value of your initial investment over time
    portfolio_value = calculate_portfolio_value(daily_returns, initial_investment)

    # Build portfolio description string
    allocation_str = ", ".join([f"{t}:{w * 100:.0f}%" for t, w in zip(tickers, weights)])
    legend_text = f"Initial: ${initial_investment:,}\nAllocation: {allocation_str}"

    # Plot the portfolio value
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_value, label=legend_text, color="blue")
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.grid(True)
    plt.legend(loc="upper left", frameon=True)
    plt.show()

    # Calculate the drawdown
    drawdown_series, max_drawdown = calculate_drawdown(portfolio_value)

    # Print the maximum drawdown value
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    # Plot the drawdown
    plt.figure(figsize=(10, 6))
    plt.plot(drawdown_series)
    plt.title('Portfolio Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.show()

def demo_volatility_sharpe():
    tickers = ['VTI',   # This single ETF gives you access to the entire US stock market, including small, mid, and large-cap companies
               'BND',    # This ETF holds a vast array of US investment-grade bonds, providing stability and income to your portfolio.
               'AGG',   #  bonds
               ]
    # Portfolio allocations
    portfolios = {
        "60/40": {"weights": [0.6, 0.2, 0.2]},
        "80/20": {"weights": [0.8, 0.1, 0.1]}
    }
    initial_investment = 10000

    # Set the start date for your investment
    start_date = '2024-01-01'

    # Fetch the data
    portfolio_data = get_historical_data(tickers, start_date=start_date)

    # --- Simulate portfolio values ---
    for name, data in portfolios.items():

        # Calculate daily returns
        daily_returns, _ = calculate_portfolio_returns(portfolio_data, data["weights"])

        avg_daily_return = np.mean(daily_returns)
        avg_annual_return = (1 + avg_daily_return) ** 252 - 1
        print(f"Average Annual Return: {avg_annual_return * 100:.2f}%")

        volatility = calculate_volatility(daily_returns)
        print(f"Annualized Volatility: {volatility:.2f}%")

        risk_free_rate = 0.01
        sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate)
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Calculate the value of your initial investment over time
        portfolio_value = calculate_portfolio_value(daily_returns, initial_investment)


        data["values"] = portfolio_value
        data["volatility"] = volatility
        data["sharpe_ratio"] = sharpe_ratio

    plt.figure(figsize=(12, 6))
    for name, data in portfolios.items():
        legend_text = (f"{name} | Vol: {data['volatility']:.1f}% | "
                       f"Sharpe: {data['sharpe_ratio']:.2f}")
        plt.plot(data["values"], label=legend_text)

    plt.title("Simulated Portfolio Growth ($10,000 Initial Investment)")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value ($)")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.grid(True)
    plt.legend(loc="upper left", frameon=True)
    plt.show()


def demo_correlation():
    # Define your portfolio and a specific investment amount
    tickers = ['VTI', 'VXUS', 'BND']
    # Set the start date for your investment
    start_date = '2024-01-01'

    # Fetch the data
    portfolio_data = get_historical_data(tickers, start_date=start_date)
    # Calculate the correlation matrix
    correlation = calculate_correlation(portfolio_data)

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Correlation Heatmap')
    plt.xticks(range(len(correlation.columns)), correlation.columns)
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.show()

def demo_montecarlo():
    # Define your portfolio and a specific investment amount
    tickers = ['VTI', 'VXUS', 'BND']
    weights = [0.60, 0.20, 0.20]  # A sample 60/20/20 allocation
    initial_investment = 10000

    # Set the start date for your investment
    start_date = '2024-01-01'

    # Fetch the data
    portfolio_data = get_historical_data(tickers, start_date=start_date)

    # Calculate daily returns
    daily_returns, _ = calculate_portfolio_returns(portfolio_data, weights)

    # Calculate the value of your initial investment over time
    portfolio_value = calculate_portfolio_value(daily_returns, initial_investment)

    # Build portfolio description string
    allocation_str = ", ".join([f"{t}:{w * 100:.0f}%" for t, w in zip(tickers, weights)])
    legend_text = f"Initial: ${initial_investment:,}\nAllocation: {allocation_str}"

    # Plot the portfolio value
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_value, label=legend_text, color="blue")
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.grid(True)
    plt.legend(loc="upper left", frameon=True)
    plt.show()

    # Calculate the drawdown
    drawdown_series, max_drawdown = calculate_drawdown(portfolio_value)

    # Print the maximum drawdown value
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    # Plot the drawdown
    plt.figure(figsize=(10, 6))
    plt.plot(drawdown_series)
    plt.title('Portfolio Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.show()

    num_years = 5
    # Run the simulation for 100 scenarios over X years
    simulated_paths = monte_carlo_simulation(daily_returns, num_simulations=100, investment_horizon_years=num_years)

    # Calculate the 5th, 50th, and 95th percentiles
    best_case = simulated_paths.iloc[-1].quantile(0.95)
    average_case = simulated_paths.iloc[-1].quantile(0.50)
    worst_case = simulated_paths.iloc[-1].quantile(0.05)

    print(f"After {num_years} years:")
    print(f"  - Best Case (95th Percentile): {best_case:.2f}x initial value")
    print(f"  - Average Case (50th Percentile): {average_case:.2f}x initial value")
    print(f"  - Worst Case (5th Percentile): {worst_case:.2f}x initial value")

    # Plot the simulated paths
    plt.figure(figsize=(10, 6))
    plt.plot(simulated_paths)
    plt.title('Monte Carlo Simulation of Future Returns')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Return')
    plt.show()

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

    if portfolio_data  is not None and not portfolio_data .empty:
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

    # Step 1: Define your portfolio and target weights
    us_tickers = ['VTI', 'VXUS', 'BND']
    weights = [0.60, 0.20, 0.20]  # A sample 60/20/20 allocation
    rebalance_tolerance = 0.05  # 5% tolerance

    start_date = '2024-09-01'

    portfolio_data_tolerance = get_historical_data(us_tickers, start_date=start_date)

    if portfolio_data_tolerance is not None and not portfolio_data_tolerance.empty:
        simple_rebalance_strategy(portfolio_data_tolerance, weights, rebalance_tolerance)



if __name__ == "__main__":

    # Example 1
    #demo1()

    #Example 2
    #demo_volatility_sharpe()

    #demo_montecarlo()

    # Example Correlation
    #demo_correlation()

    # Example rebalnce US portfolio
    # demo_time_based_rebalancing()
    # demo_tolerance_based_rebalancing()

    #



    # --- Example Usage for an Italian Portfolio ---
    # NOTE: You'll need to use specific Borsa Italiana ticker symbols.
    # E.g., EUNL.MI for iShares Core MSCI World UCITS ETF.

    it_tickers = ['EUNL.MI', 'AGGH.MI']
    it_weights = [0.70, 0.30]  # Sample 70/30 allocation

    it_portfolio_data = get_historical_data(it_tickers)

    if it_portfolio_data is not None and not it_portfolio_data.empty:
        it_daily_returns, it_cumulative_returns = calculate_portfolio_returns(it_portfolio_data, it_weights)

        print("\n--- Italian Portfolio Performance ---")
        print("Daily Returns (First 5 Days):")
        print(it_daily_returns.head())
        print("\nCumulative Returns (Last 5 Days):")
        print(it_cumulative_returns.tail())


