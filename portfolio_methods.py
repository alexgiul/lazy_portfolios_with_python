# -*- coding: utf-8 -*-

"""
This script provides the core Python functions for a book on ETF investing.
It demonstrates how to fetch historical price data, calculate portfolio performance,
analyze correlation, and simulate a simple rebalancing strategy.

Libraries required:
- pandas
- numpy
- yfinance
- openpyxl (for exporting to Excel)

To install: pip install pandas numpy yfinance matplotlib seaborn openpyxl
"""
from typing import List

import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import os
import pickle

def fetch_data(tickers, start_date, end_date, cache_dir="cache"):
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Build filename based on tickers and date range
    ticker_str = "_".join(sorted(tickers if isinstance(tickers, (list, tuple)) else [tickers]))
    cache_file = os.path.join(cache_dir, f"{ticker_str}_{start_date}_{end_date}.pkl")

    # Try loading from cache
    if os.path.exists(cache_file):
        print(f"Loading data from local cache: {cache_file}...")
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            # This prevents miscalculation when applying weights later in the script.
            if isinstance(data, pd.Series):
                # If only one ticker is fetched, it's returned as a Series, so we convert it to a DataFrame.
                data = data.to_frame()
            data = data.reindex(columns=tickers)

            return data
        except Exception as e:
            print(f"Error loading cached data: {e}, refetching from yfinance...")

    # If not found or failed, fetch from yfinance
    print(f"Fetching data for tickers: {tickers} from {start_date} to {end_date}...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
        # This prevents miscalculation when applying weights later in the script.
        if isinstance(data, pd.Series):
            # If only one ticker is fetched, it's returned as a Series, so we convert it to a DataFrame.
            data = data.to_frame()
        data = data.reindex(columns=tickers)


        # Save to cache
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        print(f"Data saved to cache: {cache_file}")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def get_historical_data(tickers:List[str], start_date:str=None, end_date:str=None):
    """
    Fetches historical adjusted close price data for a list of tickers.

    Args:
        tickers (list): A list of ticker symbols (e.g., ['VTI', 'VXUS', 'BND']).
        start_date (str): The start date in 'YYYY-MM-DD' format. Defaults to 5 years ago.
        end_date (str): The end date in 'YYYY-MM-DD' format. Defaults to today.

    Returns:
        pd.DataFrame: A DataFrame with the adjusted close prices.
    """
    if start_date is None:
        start_date = (dt.date.today() - dt.timedelta(days=5 * 365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = dt.date.today().strftime('%Y-%m-%d')

    print(f"Fetching data for tickers: {tickers} from {start_date} to {end_date}...")
    try:
        data = fetch_data(tickers, start_date, end_date)
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def calculate_portfolio_returns(data, weights: List):
    """
    Calculates the daily returns and cumulative returns of a portfolio.

    Args:
        data (pd.DataFrame): DataFrame of adjusted close prices for each asset.
        weights (list): A list of portfolio weights corresponding to the assets.

    Returns:
        tuple: A tuple containing the daily returns Series and cumulative returns Series.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        print("Data is empty or not a DataFrame.")
        return None, None

    # Calculate daily percentage returns for each asset
    daily_returns = data.pct_change().dropna()

    # Calculate the portfolio's daily returns
    portfolio_daily_returns = (daily_returns * weights).sum(axis=1)

    # Calculate the cumulative returns
    cumulative_returns = (1 + portfolio_daily_returns).cumprod()

    return portfolio_daily_returns, cumulative_returns


def calculate_portfolio_value(daily_returns, initial_investments:float):
    """
    Calculates the daily dollar value of a portfolio based on initial investments.

    Args:
        daily_returns (pd.Series): A Series of the portfolio's daily returns.
        initial_investments (float): The total initial investment amount.

    Returns:
        pd.Series: A Series with the daily dollar value of the portfolio.
    """
    if not isinstance(daily_returns, pd.Series) or daily_returns.empty:
        print("Daily returns data is empty or not a Series.")
        return None

    portfolio_value = (1 + daily_returns).cumprod() * initial_investments
    return portfolio_value


def calculate_drawdown(portfolio_value):
    """
    Calculates the drawdown and maximum drawdown of a portfolio.

    Args:
        portfolio_value (pd.Series): A Series with the daily dollar value of the portfolio.

    Returns:
        tuple: A tuple containing the drawdown Series and the max drawdown value.
    """
    if not isinstance(portfolio_value, pd.Series) or portfolio_value.empty:
        print("Portfolio value data is empty or not a Series.")
        return None

    # Calculate the running maximum (peak) of the portfolio
    running_max = portfolio_value.cummax()
    # Calculate the drawdown as the percentage decline from the peak
    drawdown = (portfolio_value - running_max) / running_max
    # Find the maximum drawdown
    max_drawdown = drawdown.min()

    return drawdown, max_drawdown


def monte_carlo_simulation(daily_returns, num_simulations:int =1000, investment_horizon_years:int=10):
    """
    Performs a Monte Carlo simulation to project future portfolio returns.

    Args:
        daily_returns (pd.Series): Historical daily returns of the portfolio.
        num_simulations (int): The number of future scenarios to simulate.
        investment_horizon_years (int): The number of years to simulate into the future.

    Returns:
        pd.DataFrame: A DataFrame containing the simulated returns for each scenario.
    """
    if not isinstance(daily_returns, pd.Series) or daily_returns.empty:
        print("Daily returns data is empty or not a Series.")
        return None

    # Calculate key metrics from historical data
    avg_daily_return = daily_returns.mean()
    std_dev_daily_return = daily_returns.std()

    # The number of trading days in the future
    trading_days = investment_horizon_years * 252

    # Generate random returns for each simulation
    simulated_returns = np.zeros((trading_days, num_simulations))
    for i in range(num_simulations):
        # Generate random numbers from a normal distribution based on historical mean and std dev
        random_returns = np.random.normal(avg_daily_return, std_dev_daily_return, trading_days)
        simulated_returns[:, i] = np.cumprod(1 + random_returns)

    return pd.DataFrame(simulated_returns)


def calculate_correlation(data):
    """
    Calculates the correlation matrix of the assets in the portfolio.

    Args:
        data (pd.DataFrame): DataFrame of adjusted close prices.

    Returns:
        pd.DataFrame: The correlation matrix.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        print("Data is empty or not a DataFrame.")
        return None

    # Calculate daily percentage returns for each asset
    daily_returns = data.pct_change().dropna()

    # Calculate the correlation matrix
    correlation_matrix = daily_returns.corr()
    return correlation_matrix


def calculate_volatility(daily_returns):
    """
    Calculates the annualized volatility (standard deviation) of a portfolio.

    Args:
        daily_returns (pd.Series): A Series of the portfolio's daily returns.

    Returns:
        float: The annualized volatility as a percentage.
    """
    if not isinstance(daily_returns, pd.Series) or daily_returns.empty:
        print("Daily returns data is empty or not a Series.")
        return None

    # Calculate standard deviation of daily returns and annualize it
    volatility = daily_returns.std() * np.sqrt(252)  # 252 trading days in a year
    return volatility * 100


def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.01):
    """
    Calculates the Sharpe Ratio for a portfolio.

    Args:
        daily_returns (pd.Series): A Series of the portfolio's daily returns.
        risk_free_rate (float): The annualized risk-free rate (e.g., 0.01 for 1%).

    Returns:
        float: The Sharpe Ratio.
    """
    if not isinstance(daily_returns, pd.Series) or daily_returns.empty:
        print("Daily returns data is empty or not a Series.")
        return None

    # Annualize the average daily return
    annualized_return = daily_returns.mean() * 252

    # Calculate the annualized volatility
    volatility = daily_returns.std() * np.sqrt(252)

    # Calculate the Sharpe Ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility
    return sharpe_ratio


def check_rebalance_at_date(data, target_weights, check_date:str, total_value):
    """
    Performs a time-based rebalancing check on a specific date.

    Args:
        data (pd.DataFrame): DataFrame of adjusted close prices.
        target_weights (list): A list of the target weights for each asset.
        check_date (str): The date to check for rebalancing in 'YYYY-MM-DD' format.
        total_value (float): The total portfolio value at the time of rebalancing.

    Returns:
        dict: A dictionary of rebalancing recommendations for each asset.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        print("Data is empty or not a DataFrame.")
        return {}

    try:
        # Use .loc to select all data up to the check_date and then get the last entry
        latest_prices = data.loc[:check_date].iloc[-1]
    except IndexError:
        print(f"No data available on or before {check_date}. Please check the date range.")
        return {}
    except KeyError:
        print(f"No data available on or before {check_date}. Please check the date range.")
        return {}

    current_weights = latest_prices / latest_prices.sum()

    rebalance_recommendations = {}
    should_rebalance = False

    print(f"\nWeight at {check_date} before rebalancing")
    print(current_weights)

    for ticker, weight in current_weights.items():
        target_weight = target_weights[list(current_weights.keys()).index(ticker)]
        if abs(weight - target_weight) > 0.001:  # Simple check for any drift
            should_rebalance = True
            rebalance_amount = (target_weight * total_value) - (weight * total_value)
            rebalance_recommendations[ticker] = rebalance_amount

    if should_rebalance:
        print(f"\nRebalancing is recommended for {check_date}!")
        for ticker, amount in rebalance_recommendations.items():
            if amount > 0:
                print(f"  - BUY {ticker}: ${amount:.2f}")
            else:
                print(f"  - SELL {ticker}: ${abs(amount):.2f}")
    else:
        print(f"\nNo rebalancing needed on {check_date}.")

    return rebalance_recommendations


def simple_rebalance_strategy(data, target_weights, tolerance):
    """
    Simulates a tolerance-based rebalancing strategy. It checks if any asset's
    weight has deviated beyond the specified tolerance and recommends a rebalance.

    Args:
        data (pd.DataFrame): DataFrame of adjusted close prices.
        target_weights (list): A list of the target weights for each asset.
        tolerance (float): The percentage deviation from the target weight
                           that triggers a rebalance (e.g., 0.05 for 5%).

    Returns:
        dict: A dictionary of rebalancing recommendations for each asset.
        dict: A dictionary of current weight for each asset.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        print("Data is empty or not a DataFrame.")
        return {}

    # Get the most recent prices
    latest_prices = data.iloc[-1]

    # Calculate the current market value of each asset based on their latest price
    current_market_values = latest_prices
    total_portfolio_value = current_market_values.sum()
    current_weights = current_market_values / total_portfolio_value


    rebalance_recommendations = {}
    should_rebalance = False

    for ticker, weight in current_weights.items():
        target_weight = target_weights[list(current_weights.keys()).index(ticker)]
        # Check if current weight deviates more than the tolerance
        if abs(weight - target_weight) > tolerance:
            should_rebalance = True
            rebalance_amount = (target_weight * total_portfolio_value) - (weight * total_portfolio_value)
            rebalance_recommendations[ticker] = rebalance_amount

    if should_rebalance:
        print("\nCurrent weights:")
        for ticker, weight in current_weights.to_dict().items():
            print(f"  - {ticker}: {weight*100:.1f}%")

        print("\nRebalancing is recommended!")
        for ticker, amount in rebalance_recommendations.items():
            if amount > 0:
                print(f"  - BUY {ticker}: ${amount:.2f}")
            else:
                print(f"  - SELL {ticker}: ${abs(amount):.2f}")
    else:
        print("\nNo rebalancing needed. Portfolio is within tolerance.")

    return rebalance_recommendations, current_weights.to_dict()


def calculate_portfolio_value_on_date(portfolio_data, initial_investment, start_date, weights):
    """
    Calculates the portfolio value on a specific end date given a start date and initial investment.

    Args:
        portfolio_data (pd.DataFrame): DataFrame of adjusted close prices for each asset.
        initial_investment (float): The initial investment amount.
        start_date (str): The date the investment was made.
        weights (list): A list of portfolio weights corresponding to the assets.

    Returns:
        float: The total value of the portfolio on the end date.
    """
    # Filter data to the relevant period
    investment_period_data = portfolio_data.loc[start_date:]
    if investment_period_data.empty:
        print(f"No data found from {start_date} .")
        return None

    # Calculate daily returns
    daily_returns = investment_period_data.pct_change().dropna()

    # Calculate portfolio returns based on the specific weights
    portfolio_daily_returns = (daily_returns * weights).sum(axis=1)

    # Calculate portfolio value
    portfolio_value = (1 + portfolio_daily_returns).cumprod() * initial_investment

    return portfolio_value.iloc[-1] if not portfolio_value.empty else None

# Chart formatter function
def thousands_formatter(x, pos):
    if x >= 1000:
        return f"{x/1000:.1f}k".rstrip('0').rstrip('.')
    return str(x)
