import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
import os

import config

def normalize_prices(data,ticker):
        """
        Normalizes numerical columns using Min-Max scaling.
        """
        df = data.copy()
        columns_to_normalize = [
        (ticker, 'Open'),
        (ticker, 'High'),
        (ticker, 'Low'),
        (ticker, 'Close'),
        (ticker, 'Adj Close')
    ]
        sc = MinMaxScaler(feature_range=(0,1))
        df[columns_to_normalize] = sc.fit_transform(df[columns_to_normalize])

        # Convert back to DataFrame for easier handling
        return df, sc
        
       
def daily_returns(data):
        """
        Computes daily returns for each ticker
        """
        df = data.copy()
        # Calculate returns for ALL tickers
        returns_dict = {}
        for ticker in config.TICKERS:
            returns_dict[ticker] = df[(ticker, 'Adj Close')].pct_change()
        
        # Create DataFrame from dictionary
        returns_df = pd.DataFrame(returns_dict)
        
        # Remove NaN rows (first row for each ticker)
        returns_df = returns_df.dropna()
        
        return returns_df

def normalize_returns_maxabs(returns_data):
    """
    Normalize returns while PRESERVING negative values.
    Scales to [-1, 1] range.
    """
    scaler = MaxAbsScaler()
    scaled = scaler.fit_transform(returns_data)
    
    scaled_df = pd.DataFrame(
        scaled,
        columns=returns_data.columns,
        index=returns_data.index
    )
    
    return scaled_df, scaler


def moving_average_50(data):
    """Compute 50-day moving average for each ticker."""
    df = data.copy()
    
    for ticker in config.TICKERS:
        df[(ticker, 'MA_50')] = df[(ticker, 'Adj Close')].rolling(
            window=50
        ).mean()
        df = df.dropna(subset=[(ticker, 'MA_50')])

    print(f"Dropped {len(data) - len(df)} rows with NaN values")
    return df[[(ticker, 'MA_50')
                for ticker in config.TICKERS]]

def moving_averages_200(data):
    """Compute 200-day moving average for each ticker."""
    df = data.copy()
    
    for ticker in config.TICKERS:
        df[(ticker, 'MA_200')] = df[(ticker, 'Adj Close')].rolling(
            window=200
        ).mean()
        df = df.dropna(subset=[(ticker, 'MA_200')])

    print(f"Dropped {len(data) - len(df)} rows with NaN values")
    return df[[(ticker, 'MA_200')
                for ticker in config.TICKERS]]