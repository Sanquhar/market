from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import xgboost as xgb

class RNN_regression:

    def __init__(self, lookback: int = 20) -> None:
        """
        Initialize the RNN_regression class with default parameters.

        Parameters:
        - lookback: int, number of past timesteps used to predict the next value.
        """
        self.lookback = lookback
        self.model = None  # Will be defined in fit()

    def fit(self, X, Y):
        """
        Fit the RNN model to the training data.

        Parameters:
        - X: numpy array of shape (n_samples, lookback * n_features)
             Example: (1000, 220) if lookback=20 and n_features=11
        - Y: numpy array of shape (n_samples,)
        """
        n_samples, total_features = X.shape
        n_features = total_features // self.lookback
        X_reshaped = X.reshape(n_samples, self.lookback, n_features)

        # Define model now that we know the number of features
        self.model = Sequential([
            LSTM(64, input_shape=(self.lookback, n_features)),
            Dense(1)
        ])
        self.model.compile(optimizer=Adam(), loss='mse')

        self.model.fit(X_reshaped, Y, epochs=20, batch_size=32, verbose=0)

    def predict(self, X):
        """
        Predict future values using the trained model.

        Parameters:
        - X: numpy array of shape (n_samples, lookback * n_features)

        Returns:
        - numpy array of shape (n_samples,) with predicted values
        """
        n_samples, total_features = X.shape
        n_features = total_features // self.lookback
        X_reshaped = X.reshape(n_samples, self.lookback, n_features)
        return self.model.predict(X_reshaped).flatten()

    def score(self, X, Y):
        """
        Compute the root mean squared error (RMSE) of the model.

        Parameters:
        - X: numpy array of shape (n_samples, lookback * n_features)
        - Y: numpy array of shape (n_samples,)

        Returns:
        - float, RMSE score
        """
        predictions = self.predict(X)
        rmse = np.sqrt(mean_squared_error(Y, predictions))
        return rmse

class XGBoost_regression:

    def __init__(self, lookback: int = 20) -> None:
        """
        Initialize the XGBoost_regression class with default parameters.

        Parameters:
        - lookback: int, number of past timesteps used to predict the next value.
        """
        self.lookback = lookback
        self.model = None  # Will be defined in fit()

    def fit(self, X, Y):
        """
        Fit the XGBoost model to the training data.

        Parameters:
        - X: numpy array of shape (n_samples, lookback * n_features)
             Example: (1000, 220) if lookback=20 and n_features=11
        - Y: numpy array of shape (n_samples,)
        """
        n_samples, total_features = X.shape
        n_features = total_features // self.lookback
        X_reshaped = X.reshape(n_samples, self.lookback * n_features)

        # Define model now that we know the number of features
        self.model = xgb.XGBRegressor(objective='reg:squarederror', 
                                      n_estimators=100, 
                                      learning_rate=0.1, 
                                      max_depth=6)
        
        self.model.fit(X_reshaped, Y)

    def predict(self, X):
        """
        Predict future values using the trained XGBoost model.

        Parameters:
        - X: numpy array of shape (n_samples, lookback * n_features)

        Returns:
        - numpy array of shape (n_samples,) with predicted values
        """
        n_samples, total_features = X.shape
        n_features = total_features // self.lookback
        X_reshaped = X.reshape(n_samples, self.lookback * n_features)
        return self.model.predict(X_reshaped)

    def score(self, X, Y):
        """
        Compute the root mean squared error (RMSE) of the XGBoost model.

        Parameters:
        - X: numpy array of shape (n_samples, lookback * n_features)
        - Y: numpy array of shape (n_samples,)

        Returns:
        - float, RMSE score
        """
        predictions = self.predict(X)
        rmse = np.sqrt(mean_squared_error(Y, predictions))
        return rmse

class MLP_regression:
    def __init__(
        self,
        lookback: int = 20,
    ) -> None:
        """
        Initialize the MLP regression model with two hidden layers of 128 neurons each.
        """
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 128),
        )

    def fit(self, X, Y):
        """
        Fit the model to the training data.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)
             Example: (1000, 220) if lookback=20 and you use 11 features per time step.
        - Y: numpy array of shape (n_samples,)
             Example: (1000,) corresponding to the target 'Close' prices.
        """
        self.model.fit(X, Y)

    def predict(self, X):
        """
        Predict target values for input features.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)

        Returns:
        - predictions: numpy array of shape (n_samples,)
        """
        return self.model.predict(X)

    def score(self, X, Y):
        """
        Compute the RMSE (Root Mean Squared Error) between predicted and true values.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        - Y: numpy array of shape (n_samples,)

        Returns:
        - rmse: float, root mean squared error
        """
        predictions = self.predict(X)
        rmse = np.sqrt(mean_squared_error(Y, predictions))
        return rmse       