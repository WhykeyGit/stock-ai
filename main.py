
import config
from stock_data import StockData
from lstm import LSTMModel
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# The tickers to fetch data for
tickers = config.TICKERS
start_date = config.START_DATE
end_date = config.END_DATE
models = {}


def main():
    # ticker = "NVDA"
    # ygf = yf.download(ticker, start="2025-01-01", end="2025-10-01", auto_adjust=False)
    # print(ygf.columns)
    # sc = MinMaxScaler(feature_range=(0,1))
    # normalized_nvda_2025 = sc.fit_transform(ygf)
    # normalized_nvda_2025 = pd.DataFrame(normalized_nvda_2025, columns=ygf.columns)
    # print(normalized_nvda_2025.head())


    for ticker in config.TICKERS:
        model = LSTMModel(ticker=ticker, sequence_length=1)
        print(model.x_train.shape, model.y_train.shape, model.x_test.shape, model.y_test.shape)
        model.build_lstm_model(units=150, dropout_rate=0.3)
        model.train_lstm(epochs=50, batch_size=32)
        models[ticker] = model
    
    # models["NVDA"].evaluate()
    # print(f"NVDA predicted normalized close price for 03-01-2025: {tomorrow}")
    # print(f"NVDA actual normalized close price for 03-01-2025: {normalized_nvda_2025[('Close','NVDA')].iloc[2]}")



if __name__ == "__main__":
    main()