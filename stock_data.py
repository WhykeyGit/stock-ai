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
from sklearn.preprocessing import MinMaxScaler
import os

import config



class StockData:
    def __init__(self, ticker):
        self.tickers = config.TICKERS
        self.ticker = ticker
        self.start =config.START_DATE
        self.end = config.END_DATE

        self.data = None
        # Data with Daily returns, Rolling volatility, Cumulative returns, Moving averages
        self.processed_data = None
        self.normalized_data = None
        self.data_with_target = None

    def download(self, use_cache=True, force_refresh=False, cache_path=None):
        """
        Downloads historical stock data for the specified tickers and date range.
        
        Args:
            use_cache: Whether to use caching (load and save)
            force_refresh: Force fresh download even if cache exists
            cache_path: Custom cache file path (defaults to config.DATA_CACHE_PATH)
        """
        cache_file = cache_path or config.DATA_CACHE_PATH
        
        # Try loading from cache
        if use_cache and not force_refresh and os.path.exists(cache_file):
            try:
                print(f"Loading cached data from {cache_file}...")
                df = pd.read_parquet(cache_file)
                self.data = df
                self.processed_data = df.copy()
                print("Loaded from cache")
                return
            except Exception as e:
                print(f"Cache load failed ({e}). Downloading fresh data...")
        
        # Download fresh data
        print(f"Downloading data for {self.tickers} from {self.start} to {self.end}...")
        try:
            df = yf.download(
                self.tickers, 
                start=self.start, 
                end=self.end, 
                group_by="ticker"
            )
            
            # Normalize structure for single tickers
            if isinstance(self.tickers, str) or len(self.tickers) == 1:
                ticker = self.tickers if isinstance(self.tickers, str) else self.tickers[0]
                df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
            
            df.reset_index(inplace=True)
            self.data = df
            self.processed_data = df.copy()
            print(f"Downloaded {len(df)} rows")
            
        except Exception as e:
            print(f"Download failed: {e}")
            raise
        
        # Save to cache
        if use_cache:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                df.to_parquet(cache_file)
                print(f"Data cached at {cache_file}")
            except Exception as e:
                print(f"Cache save failed: {e}")

    # ==============================
    # Feature Engineering
    # ==============================

    def normalize(self,data):
        """
        Normalizes numerical columns using Min-Max scaling.
        """
        df = data.copy()
        sc = MinMaxScaler(feature_range=(0,1))
        self.normalized_data = sc.fit_transform(df.drop(columns = ['Date']))

        # Convert back to DataFrame for easier handling
        self.normalized_data = pd.DataFrame(self.normalized_data, columns=df.columns[1:])
        
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
            df['Daily Return'] = close.pct_change()
            # Rolling volatility (annualized)
            df[f'{ticker}_Volatility'] = (
                df['Daily Return'].rolling(window=config.VOLATILITY_WINDOW).std()
                * np.sqrt(config.TRADING_DAYS_PER_YEAR)
            )
            # Cumulative return
            df[f'{ticker}_CumulativeReturn'] = (1 + df['Daily Return']).cumprod() - 1

            # Moving averages
            for window in config.MOVING_AVERAGE_WINDOWS:
                df[f'{ticker}_MA{window}'] = close.rolling(window=window).mean()

        self.processed_data = df


    # Function to concatenate the date, stock price, and volume in one dataframe
        # Create a trading window for the next n days
    def trading_close_price_target_window(self):

        self.normalize(self.data)
        # 1 day window
        n = 1
        data = self.normalized_data.copy()  
        for i in self.tickers:
            print(i)
            data = data.drop(columns=[(i,'Open'),(i,'High'),(i,'Volume'),(i,'Low')])

        # Create a column containing the prices for the next 1 days
        for i in self.tickers:
            data[(i,'Target')] = data[(i,"Close")].shift(-n)
        
        # return the new dataset
        self.data_with_target = data[:-1].sort_index(level='Ticker', axis=1)


    # Create a trading window of the daily return for the next n days
    def trading_daily_return_target_window(self):

        # 1 day window
        n = 1
        data = self.normalized_data.copy()
        for i in self.tickers:
            print(i)
            data = data.drop(columns=[(i,'Open'),(i,'High'),(i,'Volume'),(i,'Low')])

        # Create a column containing the prices for the next 1 days
        for i in self.tickers:
            data[(i,'Target')] = data[(i,"Daily Return")].shift(-n)
        
        # return the new dataset
        self.data_with_target = data[:-1].sort_index(level='Ticker', axis=1)


    # ==============================
    # train-test split
    # ==============================
    def train_test_split(self, test_size=0.2, sequence_length=1):
        """
        Split data for a SINGLE ticker only.
        """
        df = self.data_with_target.copy()
        split_idx = int(len(df) * (1 - test_size))
        
        # Extract ONLY this ticker's data
        close_prices = df[(self.ticker, 'Close')].values
        target_prices = df[(self.ticker, 'Target')].values
        
        # Split into train and test
        train_close = close_prices[:split_idx]
        train_target = target_prices[:split_idx]
        test_close = close_prices[split_idx:]
        test_target = target_prices[split_idx:]
        
        # Create sequences
        X_train_list = []
        y_train_list = []
        for i in range(sequence_length, len(train_close)):
            X_train_list.append(train_close[i-sequence_length:i])
            y_train_list.append(train_target[i])
        
        X_test_list = []
        y_test_list = []
        for i in range(sequence_length, len(test_close)):
            X_test_list.append(test_close[i-sequence_length:i])
            y_test_list.append(test_target[i])
        
        # Convert to arrays
        X_train = np.array(X_train_list).reshape(-1, sequence_length, 1)
        y_train = np.array(y_train_list)
        X_test = np.array(X_test_list).reshape(-1, sequence_length, 1)
        y_test = np.array(y_test_list)
        
        print(f"Training set for {self.ticker}: X shape {X_train.shape}, y shape {y_train.shape}")
        print(f"Testing set for {self.ticker}: X shape {X_test.shape}, y shape {y_test.shape}")
        
        return X_train, y_train, X_test, y_test

    # ==============================
    # Visualization
    # ==============================

    # def plot_moving_averages(self):
    #     """
    #     Plots closing prices with moving averages using Matplotlib.
    #     """
    #     if self.processed_data is None:
    #         raise ValueError("Indicators must be added before plotting moving averages.")

    #     plt.figure(figsize=config.FIGURE_SIZE)
    #     for ticker in self.tickers:
    #         plt.plot(self.processed_data['Date'], self.processed_data[ticker]['Close'], label=f'{ticker} Closing Price')
    #         for window in config.MOVING_AVERAGE_WINDOWS:
    #             plt.plot(
    #                 self.processed_data['Date'],
    #                 self.processed_data[f'{ticker}_MA{window}'],
    #                 label=f'{ticker} {window}-Day MA'
    #             )

    #     plt.title('Stock Prices with Moving Averages')
    #     plt.xlabel('Date')
    #     plt.ylabel('Price (USD)')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

