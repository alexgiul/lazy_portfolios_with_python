import matplotlib.pyplot as plt
from portfolio_methods import *
from matplotlib.ticker import FuncFormatter

if __name__ == "__main__":
    tickers = ['VTI', 'BND', 'AGG']
    # Portfolio allocations
    portfolios = {
        "60/40": {"weights": [0.6, 0.2, 0.2]},
        "80/20": {"weights": [0.8, 0.1, 0.1]}
    }
    initial_investment = 10000

    # Set the start date for your investment
    start_date = '2020-01-01'

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
