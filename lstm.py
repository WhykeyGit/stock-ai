from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tensorflow import keras
import pandas as pd
import numpy as np
import config
from stock_data import StockData

class LSTMModel:

    def __init__(self, ticker, sequence_length=1):
        self.keras_model = None
        self.trained = False
        self.tickers = config.TICKERS
        self.ticker = ticker
        self.sequence_length = sequence_length

        # Prepare data
        sd = StockData(ticker=self.ticker)
        sd.download()
        self.data = sd.data

        # Split data into training and testing sets for daily_return
        self.x_train, self.y_train, self.x_test, self.y_test = sd.train_test_split(test_size=0.2, sequence_length=1, column_name='Daily Return', data=self.daily_return_target)


    def build_lstm_model(self, units=150, dropout_rate=0.3):
        """
        Build the LSTM model architecture.
        
        Args:
            units: Number of LSTM units per layer (default 150)
            dropout_rate: Dropout rate for regularization (default 0.3)
        """
        # Input shape: (timesteps, features)
        inputs = keras.layers.Input(shape=(self.x_train.shape[1], self.x_train.shape[2]))
        
        # First LSTM layer with return_sequences=True to pass to next LSTM
        x = keras.layers.LSTM(units, return_sequences=True)(inputs)
        x = keras.layers.Dropout(dropout_rate)(x)
        
        # Second LSTM layer
        x = keras.layers.LSTM(units, return_sequences=True)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        
        # Third LSTM layer - no return_sequences needed for final layer
        x = keras.layers.LSTM(units)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        
        # Output layer - linear activation for regression
        outputs = keras.layers.Dense(1, activation='linear')(x)
        
        # Create and compile model
        self.lstm_model = keras.Model(inputs=inputs, outputs=outputs)
        self.lstm_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        self.lstm_model.summary()
        
        return self.lstm_model
    
    # Train the model
    def train_lstm(self, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the LSTM model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Proportion of training data to use for validation
        """
        if self.lstm_model is None:
            print("Building model first...")
            self.build_lstm_model()
        
        print(f"\nTraining {self.ticker} LSTM model for {epochs} epochs...")
        
        # Add early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.lstm_model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.trained = True
        print("\nTraining complete!")
        
        return history


    def evaluate(self):
        """
        Evaluate the model on test data.
        """
        # Evaluate on test set
        test_loss, test_mae = self.lstm_model.evaluate(self.x_test, self.y_test, verbose=0)
        
        print(f"\nTest Results:")
        print(f"Test Loss (MSE): {test_loss:.6f}")
        print(f"Test MAE: {test_mae:.6f}")
        
        return test_loss, test_mae
    
    def predict(self, X=None):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input data (uses test set if None)
        """
        if not self.trained:
            raise Exception("Model is not trained yet.")
        
        if X is None:
            X = self.x_test
        
        predictions = self.lstm_model.predict(X)
        return predictions    
    

    def predict_next_day(self, recent_prices):
        """
        Predict the next day's closing price given recent prices.
        
        Args:
            recent_prices: Array of recent closing prices (length = sequence_length)
        """
        if not self.trained:
            raise Exception("Model is not trained yet.")
        
        # Reshape for model input: (1, sequence_length, 1)
        X_input = np.array(recent_prices).reshape(1, self.sequence_length, 1)
        
        prediction = self.lstm_model.predict(X_input, verbose=0)
        return prediction[0][0]    