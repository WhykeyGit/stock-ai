import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

import config

def normalize(data,ticker):
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
        for ticker in config.TICKERS:
            # Daily return
            df[(ticker,'Daily Return')] = df[(ticker,'Adj Close')].pct_change()

        return df[(ticker,'Daily Return')][1:]  # Skip first NaN row


def moving_averages(data):
    """
    Computes moving averages for each ticker based on config settings
    """
    df = data.copy()
    for ticker in config.TICKERS:
        for window in config.MOVING_AVERAGE_WINDOWS:
            df[(ticker,f'MA_{window}')] = df[(ticker,'Adj Close')].rolling(window=window).mean()
        
    return df