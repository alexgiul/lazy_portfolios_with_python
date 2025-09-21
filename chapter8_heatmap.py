import matplotlib.pyplot as plt
from matplotlib.pyplot import annotate

from portfolio_methods import *
import seaborn as sns

if __name__ == "__main__":
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
    plt.imshow(correlation, cmap='coolwarm')
    plt.colorbar(label='Asset Correlation Coefficient')
    plt.title('Correlation Heatmap')
    plt.xticks(range(len(correlation.columns)), correlation.columns)
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.savefig('correlation_heatmap.png')
    plt.show()