import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load historical stock data
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
# Since data is downloaded with Date as index, we reset it to make Date a column
data.reset_index(inplace=True)

# Basic data exploration
print(data.info())
print(data.describe())
print(data.head())
print(data.tail())
print(data.isnull().sum())
print(data.duplicated().sum())
print(data.columns)
print(data.nunique())
print(data.dtypes)
print(data.corr())
print(data.shape)
print(data['Close'].value_counts())
print(data['Volume'].value_counts())
print(data['Date'].value_counts())
print(data['Open'].value_counts())
print(data['High'].value_counts())
print(data['Low'].value_counts())