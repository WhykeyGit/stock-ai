
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
    ticker = "NVDA"
    ygf = yf.download(ticker, start="2025-01-01", end="2025-10-01")
    sc = MinMaxScaler(feature_range=(0,1))
    normalized_nvda_2025 = sc.fit_transform(ygf)
    normalized_nvda_2025 = pd.DataFrame(normalized_nvda_2025, columns=ygf.columns)
    print(normalized_nvda_2025.head())

    for ticker in config.TICKERS:
        model = LSTMModel(ticker=ticker, sequence_length=1)
        model.build_lstm_model(units=150, dropout_rate=0.3)
        model.train_lstm(epochs=50, batch_size=32)
        models[ticker] = model
    
    models["NVDA"].evaluate()
    tomorrow = models["NVDA"].predict_next_day([0.54])
    print(f"NVDA predicted normalized close price for 03-01-2025: {tomorrow}")
    print(f"NVDA actual normalized close price for 03-01-2025: {normalized_nvda_2025[('Close','NVDA')].iloc[2]}")

    # Build and train the LSTM model with different dropout rates
    # Df with as columns dropout rates and test MAE and MSE
    # df_different_dropout = pd.DataFrame(columns=["dropout_rate", "test_mae", "test_mse"])
    # dropout_rates = np.arange(0.1, 1, 0.1)
    # for rate in dropout_rates:
    #     print(f"\nTraining LSTM model with dropout rate: {rate}")
    #     lr.build_lstm_model(units=150, dropout_rate=rate)
    #     lr.train_lstm(epochs=50, batch_size=32)
    #     test_mse, test_mae = lr.evaluate()
    #     df_different_dropout = pd.concat([df_different_dropout, pd.DataFrame({"dropout_rate": [rate], "test_mae": [test_mae], "test_mse": [test_mse]})], ignore_index=True)
    # print("\nTest results for different dropout rates:")
    # print(df_different_dropout)


if __name__ == "__main__":
    main()