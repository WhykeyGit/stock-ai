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

def macd_single_ticker(data):

    df = data.copy()
     # Calculate the 12-period EMA
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    print(df['EMA12'].head())
    # Calculate the 26-period EMA
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    print(df['EMA26'].head())
    # Calculate MACD (the difference between 12-period EMA and 26-period EMA)
    df['MACD'] = df['EMA12'] - df['EMA26']

    # Calculate the 9-period EMA of MACD (Signal Line)
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df = df[['EMA12','EMA26','MACD','Signal_Line']]
    # Check for MACD and Signal Line crossovers in the last two rows
    last_row = df.iloc[-1]
    second_last_row = df.iloc[-2]
    if second_last_row['MACD'] > second_last_row['Signal_Line'] and last_row['MACD'] < last_row['Signal_Line']:
        print('Cross Below Signal Line')
    elif second_last_row['MACD'] < second_last_row['Signal_Line'] and last_row['MACD'] > last_row['Signal_Line']:
        print('Cross Above Signal Line')
    else:
        print('No Crossover')
    return df