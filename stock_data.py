"""
stock_data.py

Defines the StockData class and related utility functions
for downloading, processing, and visualizing stock market data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
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
        self.copy_data = None
        self.daily_returns_data = None
        self.normalized_data = None
        self.close_price_target = None
        self.daily_return_target = None

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
                self.copy_data = df.copy()
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
                group_by="ticker",
                auto_adjust=False

            )
            
            # Normalize structure for single tickers
            if isinstance(self.tickers, str) or len(self.tickers) == 1:
                ticker = self.tickers if isinstance(self.tickers, str) else self.tickers[0]
                df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
            
            df.reset_index(inplace=True)
            self.data = df
            self.copy_data = df.copy()
            print(f"Downloaded {len(df)} rows")
            
        except Exception as e:
            print(f"Download failed: {e}")
            raise
        
        # Save to cache
        if use_cache:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                df.to_parquet(cache_file)
                self.copy_data = df.copy()
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
        df = self.copy_data
        sc = MinMaxScaler(feature_range=(0,1))
        self.normalized_data = sc.fit_transform(df.drop(columns = ['Date']))

        # Convert back to DataFrame for easier handling
        self.normalized_data = pd.DataFrame(self.normalized_data, columns=df.columns[1:])
        
    def daily_returns(self):
        """
        Computes daily returns for each ticker
        """
        df = self.copy_data
        for ticker in self.tickers:
            # Daily return
            df[(ticker,'Daily Return')] = df[(ticker,'Adj Close')].pct_change()

        self.daily_returns_data = df[(ticker,'Daily Return')][1:]  # Skip first NaN row

    

    # Function to concatenate the date, stock price, and volume in one dataframe
        # Create a trading window for the next n days
    def trading_close_price_target_window(self):

        self.normalize(self.copy_data)
        # 1 day window
        n = 1
        data = self.normalized_data
        for i in self.tickers:
            print(i)
            data = data.drop(columns=[(i,'Open'),(i,'High'),(i,'Volume'),(i,'Low')])

        # Create a column containing the prices for the next 1 days
        for i in self.tickers:
            data[(i,'Target')] = data[(i,"Close")].shift(-n)
        
        # return the new dataset
        self.close_price_target = data[:-1].sort_index(level='Ticker', axis=1)


    # Create a trading window of the daily return for the next n days
    def trading_daily_return_target_window(self):

        self.daily_returns()
        # 1 day window
        n = 1
        self.normalize(data=self.daily_returns_data)
        data = self.normalized_data
        # for i in self.tickers:
        #     print(i)
        #     data = data.drop(columns=[(i,'Open'),(i,'Close'),(i,'High'),(i,'Volume'),(i,'Low')])

        # Create a column containing the prices for the next 1 days
        for i in self.tickers:
            data[(i,'Target')] = data[(i,"Daily Return")].shift(-n)
        
        # return the new dataset
        self.daily_return_target = data[:-1].sort_index(level='Ticker', axis=1)


    # ==============================
    # train-test split
    # ==============================
    def train_test_split(self, test_size=0.2, sequence_length=1,column_name=None,data=None):
        """
        Split data for a SINGLE ticker only.
        """
        df = data.copy()
        split_idx = int(len(df) * (1 - test_size))
        
        # Extract ONLY this ticker's data
        begin_prices = df[(self.ticker, column_name)].values
        target_prices = df[(self.ticker, 'Target')].values
        
        # Split into train and test
        train_close = begin_prices[:split_idx]
        train_target = target_prices[:split_idx]
        test_close = begin_prices[split_idx:]
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

