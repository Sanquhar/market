from sklearn.neural_network import MLPRegressor
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

class RNN_regression:

    def __init__(self, lookback: int = 20) -> None:
        """
        Initialize the RNN_regression class with default parameters.

        Parameters:
        - lookback: int, number of past timesteps used to predict the next value.
        """
        self.lookback = lookback
        self.model = None  # Will be defined in fit()
        self.training_time = None

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

        start = time.time()
        self.model.fit(X_reshaped, Y, epochs=20, batch_size=32, verbose=0)
        end = time.time()
        self.training_time = end - start

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

    def score(self, X, Y, y_scaler):
        """
        Compute the RMSE (Root Mean Squared Error) between predicted and true values.
        """
        predictions = self.predict(X)
        predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
        y_true = y_scaler.inverse_transform(Y.reshape(-1, 1)).ravel()
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        return rmse

    def get_mae(self, X, Y, y_scaler):
        """
        Compute the Mean Absolute Error.
        """
        predictions = self.predict(X)
        predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
        y_true = y_scaler.inverse_transform(Y.reshape(-1, 1)).ravel()
        return mean_absolute_error(y_true, predictions)

    def get_r2(self, X, Y, y_scaler):
        """
        Compute the R^2 (coefficient of determination) score.
        """
        predictions = self.predict(X)
        predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
        y_true = y_scaler.inverse_transform(Y.reshape(-1, 1)).ravel()
        return r2_score(y_true, predictions)

    def get_training_time(self):
        """
        Return the training time in seconds.
        """
        return self.training_time

    def get_nb_trainable_parameters(self):
        """
        Return the number of trainable parameters in the model.
        """
        return np.sum([np.prod(v.shape) for v in self.model.trainable_weights])
    
class XGBoost_regression:

    def __init__(self, lookback: int = 20) -> None:
        """
        Initialize the XGBoost_regression class with default parameters.

        Parameters:
        - lookback: int, number of past timesteps used to predict the next value.
        """
        self.lookback = lookback
        self.model = None  # Will be defined in fit()
        self.training_time = None

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
        
        start = time.time()
        self.model.fit(X_reshaped, Y)
        end = time.time()
        self.training_time = end - start

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

    def score(self, X, Y, y_scaler):
        """
        Compute the RMSE (Root Mean Squared Error) between predicted and true values.
        """
        predictions = self.predict(X)
        predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
        y_true = y_scaler.inverse_transform(Y.reshape(-1, 1)).ravel()
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        return rmse

    def get_mae(self, X, Y, y_scaler):
        """
        Compute the Mean Absolute Error.
        """
        predictions = self.predict(X)
        predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
        y_true = y_scaler.inverse_transform(Y.reshape(-1, 1)).ravel()
        return mean_absolute_error(y_true, predictions)

    def get_r2(self, X, Y, y_scaler):
        """
        Compute the R^2 (coefficient of determination) score.
        """
        predictions = self.predict(X)
        predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
        y_true = y_scaler.inverse_transform(Y.reshape(-1, 1)).ravel()
        return r2_score(y_true, predictions)

    def get_nb_trainable_parameters(self):
        """
        Return number of trainable parameters (approximate).
        """
        booster = self.model.get_booster()
        return sum([int(s.split('=')[1]) for s in booster.attributes().get('best_msg', '').split(',') if 'param' not in s]) if booster else 0

    def get_training_time(self):
        """
        Return training duration in seconds.
        """
        return self.training_time

class MLP_regression:
    def __init__(
        self,
        lookback: int = 20,
    ) -> None:
        """
        Initialize the MLP regression model with two hidden layers of 128 neurons each.
        """
        self.model = MLPRegressor(hidden_layer_sizes=(500, 500))
        self.training_time = None

    def fit(self, X, Y):
        """
        Fit the model to the training data and record training time.
        """
        start = time.time()
        self.model.fit(X, Y)
        end = time.time()
        self.training_time = end - start

    def predict(self, X):
        """
        Predict target values for input features.
        """
        return self.model.predict(X)

    def score(self, X, Y, y_scaler):
        """
        Compute the RMSE (Root Mean Squared Error) between predicted and true values.
        """
        predictions = self.predict(X)
        predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
        y_true = y_scaler.inverse_transform(Y.reshape(-1, 1)).ravel()
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        return rmse

    def get_mae(self, X, Y, y_scaler):
        """
        Compute the Mean Absolute Error.
        """
        predictions = self.predict(X)
        predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
        y_true = y_scaler.inverse_transform(Y.reshape(-1, 1)).ravel()
        return mean_absolute_error(y_true, predictions)

    def get_r2(self, X, Y, y_scaler):
        """
        Compute the R^2 (coefficient of determination) score.
        """
        predictions = self.predict(X)
        predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
        y_true = y_scaler.inverse_transform(Y.reshape(-1, 1)).ravel()
        return r2_score(y_true, predictions)

    def get_nb_trainable_parameters(self):
        """
        Estimate the number of trainable parameters.
        """
        total = 0
        for coef, intercept in zip(self.model.coefs_, self.model.intercepts_):
            total += coef.size + intercept.size
        return total

    def get_training_time(self):
        """
        Return the training time in seconds.
        """
        return self.training_time
