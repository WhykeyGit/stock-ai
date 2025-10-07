"""
stock_data.py

Defines the StockData class and related utility functions
for downloading, processing, and visualizing stock market data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt

import config


class StockData:
    """
    Represents a collection of stock time series data for multiple tickers.
    Handles downloading, processing, and basic visualization.
    """

    def __init__(self, tickers=None, start=None, end=None):
        self.tickers = tickers or config.TICKERS
        self.start = start or config.START_DATE
        self.end = end or config.END_DATE
        self.data = None  # raw downloaded data
        self.processed_data = None  # data with indicators added

    # ==============================
    # Data Download
    # ==============================

    def download(self, cache=False, path=None):
        """
        Downloads historical stock data for the specified tickers and date range.
        Optionally caches the data to a file.
        """
        print(f"Downloading data for {self.tickers} from {self.start} to {self.end}...")
        df = yf.download(self.tickers, start=self.start, end=self.end, group_by="ticker")
        df.reset_index(inplace=True)
        self.data = df
        self.processed_data = df.copy()

        if cache:
            save_path = path or config.DATA_CACHE_PATH
            df.to_parquet(save_path)
            print(f"✅ Data cached at {save_path}")

    # ==============================
    # 🧮 Feature Engineering
    # ==============================

    def add_indicators(self):
        """
        Computes common financial indicators for each ticker:
        - Daily returns
        - Rolling volatility
        - Cumulative returns
        - Moving averages
        """
        if self.processed_data is None:
            raise ValueError("Data must be downloaded before adding indicators.")

        df = self.processed_data

        for ticker in self.tickers:
            close = df[ticker]['Close']
            # Daily return
            df[f'{ticker}_Return'] = close.pct_change()
            # Rolling volatility (annualized)
            df[f'{ticker}_Volatility'] = (
                df[f'{ticker}_Return'].rolling(window=config.VOLATILITY_WINDOW).std()
                * np.sqrt(config.TRADING_DAYS_PER_YEAR)
            )
            # Cumulative return
            df[f'{ticker}_CumulativeReturn'] = (1 + df[f'{ticker}_Return']).cumprod() - 1

            # Moving averages
            for window in config.MOVING_AVERAGE_WINDOWS:
                df[f'{ticker}_MA{window}'] = close.rolling(window=window).mean()

        self.processed_data = df

    # ==============================
    # Visualization
    # ==============================

    def plot_moving_averages(self):
        """
        Plots closing prices with moving averages using Matplotlib.
        """
        if self.processed_data is None:
            raise ValueError("Indicators must be added before plotting moving averages.")

        plt.figure(figsize=config.FIGURE_SIZE)
        for ticker in self.tickers:
            plt.plot(self.processed_data['Date'], self.processed_data[ticker]['Close'], label=f'{ticker} Closing Price')
            for window in config.MOVING_AVERAGE_WINDOWS:
                plt.plot(
                    self.processed_data['Date'],
                    self.processed_data[f'{ticker}_MA{window}'],
                    label=f'{ticker} {window}-Day MA'
                )

        plt.title('Stock Prices with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.tight_layout()
        plt.show()

