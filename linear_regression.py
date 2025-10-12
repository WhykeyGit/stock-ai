from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tensorflow import keras
import pandas as pd
import config
from stock_data import StockData

class LRModel:

    def __init__(self):
        self.model = LinearRegression()
        sd = StockData()
        sd.download()
        sd.normalize()
        sd.trading_target_window()
        self.keras_model = None
        self.trained = False
        self.tickers = config.TICKERS
        self.data = sd.normalized_data
        self.data_with_target = sd.data_with_target

    # Train the model
    def train(self, X, y):
        self.model.fit(X, y)



    def predict(self, X):
        if not self.trained:
            raise Exception("Model is not trained yet.")
        return self.model.predict(X)
