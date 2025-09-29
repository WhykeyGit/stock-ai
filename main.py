import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# The tickers to fetch data for
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]

# Load historical stock data
data = yf.download(tickers, start="2020-01-01", end="2025-01-01", group_by='ticker')
# Since data is downloaded with Date as index, we reset it to make Date a column
data.reset_index(inplace=True)

# moving averages
plt.figure(3, figsize=(14, 7))
for ticker in tickers:
    data[f'{ticker}_MA50'] = data[ticker]['Close'].rolling(window=50).mean()
    data[f'{ticker}_MA200'] = data[ticker]['Close'].rolling(window=200).mean()
    plt.plot(data['Date'], data[ticker]['Close'], label=f'{ticker} Closing Price')
    plt.plot(data['Date'], data[f'{ticker}_MA50'], label=f'{ticker} 50-Day MA')
    plt.plot(data['Date'], data[f'{ticker}_MA200'], label=f'{ticker} 200-Day MA')
plt.title('Stock Prices with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()