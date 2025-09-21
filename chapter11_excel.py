import matplotlib.pyplot as plt
from portfolio_methods import *

if __name__ == "__main__":
    initial_investment = 10000
    start_date = '2024-01-01'

    # Assuming a 60/20/20 portfolio of VTI, VXUS, and BND
    tickers = ['VTI', 'VXUS', 'BND']
    weights = [0.60, 0.20, 0.20]

    # Fetch the data as Pandas Dataframe
    portfolio_data = get_historical_data(tickers, start_date=start_date)

    # Calculate daily returns
    daily_returns, _ = calculate_portfolio_returns(portfolio_data, weights)

    # Calculate the value of your initial investment over time
    portfolio_value = calculate_portfolio_value(daily_returns, initial_investment)

    # Calculate the drawdown
    drawdown_series, _ = calculate_drawdown(portfolio_value)

    # Calculate the correlation matrix
    correlation_matrix = calculate_correlation(portfolio_data)
    num_years = 5
    # Run the simulation for 100 scenarios over X years
    simulated_paths = monte_carlo_simulation(daily_returns, num_simulations=100, investment_horizon_years=num_years)

    risk_free_rate = 0.01
    sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate)
    volatility = calculate_volatility(daily_returns)

    # Create a single Excel file with multiple sheets
    with pd.ExcelWriter('Portfolio_Report.xlsx') as writer:
        # Sheet 1: Portfolio Performance
        portfolio_value.to_excel(writer, sheet_name='Performance')

        # Sheet 2: Drawdown Analysis
        drawdown_series.to_excel(writer, sheet_name='Drawdown')

        # Sheet 3: Asset Correlation
        correlation_matrix.to_excel(writer, sheet_name='Correlation')

        # Sheet 4: Monte Carlo Simulation Results
        simulated_paths.to_excel(writer, sheet_name='Monte Carlo')

        # Sheet 5: Portfolio Summary (key metrics)
        summary_data = pd.DataFrame({
            'Metric': ['Annualized Volatility', 'Sharpe Ratio'],
            'Value': [volatility, sharpe_ratio]
        })
        summary_data.to_excel(writer, sheet_name='Summary', index=False)


