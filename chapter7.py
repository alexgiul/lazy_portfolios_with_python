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

    # Fetch the data as Pandas Dataframe
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
