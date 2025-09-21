import matplotlib.pyplot as plt
from portfolio_methods import *

if __name__ == "__main__":

    initial_investment = 10000
    start_date = '2024-01-01'

    # Assuming a 60/20/20 portfolio of VTI, VXUS, and BND
    tickers = ['VTI', 'VXUS', 'BND']
    portfolio_weights = [0.60, 0.20, 0.20]
    # Set the start date for your investment

    # Our benchmark will be a custom mix of these same ETFs
    benchmark_tickers = ['SPY']
    benchmark_weights = [1,]

    # Fetch the data as Pandas Dataframe
    portfolio_data = get_historical_data(tickers, start_date=start_date)
    # Fetch the data as Pandas Dataframe
    benchmark_data = get_historical_data(benchmark_tickers, start_date=start_date)


    # Calculate portfolio and benchmark cumulative returns
    portfolio_daily_returns, portfolio_cumulative_returns = calculate_portfolio_returns(portfolio_data, portfolio_weights)
    benchmark_daily_returns, benchmark_cumulative_returns = calculate_portfolio_returns(benchmark_data, benchmark_weights)

    # Calculate the value of your initial investment over time
    portfolio_value = calculate_portfolio_value(portfolio_daily_returns, initial_investment)
    benchmark_value = calculate_portfolio_value(benchmark_daily_returns, initial_investment)

    # Plot the comparison
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_cumulative_returns, label='Your Portfolio')
    plt.plot(benchmark_cumulative_returns, label='Benchmark')
    plt.title('Portfolio vs. Benchmark Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Drawdown plot

    # Calculate the drawdown
    portfolio_drawdown_series, portfolio_max_drawdown = calculate_drawdown(portfolio_value)
    benchmark_drawdown_series, benchmark_max_drawdown = calculate_drawdown(benchmark_value)

    # Print the maximum drawdown value
    print(f"Portfolio Maximum Drawdown: {portfolio_max_drawdown:.2%}")
    print(f"Benchmark Maximum Drawdown: {benchmark_max_drawdown:.2%}")

    # Plot the drawdown
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_drawdown_series, label='Your Portfolio')
    plt.plot(benchmark_drawdown_series, label='Benchmark')
    plt.title('Portfolio vs. Benchmark Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate Sharpe Ratios
    portfolio_sharpe = calculate_sharpe_ratio(portfolio_daily_returns)
    benchmark_sharpe = calculate_sharpe_ratio(benchmark_daily_returns)

    print(f"Your Portfolio's Sharpe Ratio: {portfolio_sharpe:.2f}")
    print(f"Benchmark's Sharpe Ratio: {benchmark_sharpe:.2f}")
