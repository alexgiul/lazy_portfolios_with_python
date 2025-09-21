import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from portfolio_methods import *

if __name__ == "__main__":
    # Define your portfolio and a specific investment amount
    tickers = ['VTI', 'VXUS', 'BND']
    weights = [0.60, 0.20, 0.20]  # A sample 60/20/20 allocation
    initial_investment = 10000

    # Set the start date for your investment
    start_date = '2024-01-01'

    # Fetch the data as Pandas DataFrame
    portfolio_data = get_historical_data(tickers, start_date=start_date)

    # Calculate daily returns
    daily_returns, _ = calculate_portfolio_returns(portfolio_data, weights)

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
    plt.savefig('monte_carlo_simulation.png')
    plt.show()