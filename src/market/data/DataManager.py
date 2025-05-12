import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import re
from market.model.model import MLP_regression, RNN_regression, XGBoost_regression
from market.model.preprocessing import create_features
from sklearn.preprocessing import StandardScaler
import pickle

class DataManager:
    """
    DataManager handle the raw data, the features data, and the models of all stocks.
    """
    def __init__(
            self, 
            data_dir="data/stocks", 
            model_dir="generated/model",
            features_dir="generated/features"
            ):
        """
        Create directories if they don't exist.

        Args:
            data_dir (str, optional): _description_. Defaults to "data/stocks".
            model_dir (str, optional): _description_. Defaults to "generated/model".
            features_dir (str, optional): _description_. Defaults to "generated/features".
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.features_dir = features_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

    def download_data_investpy(self, ticker, start_datetime, end_datetime):
        print(f"[INFO] Downloading {ticker} data until {end_datetime}")

        start_date = (end_datetime - pd.Timedelta(days=5_000)).strftime("%d/%m/%Y")
        end_date = (end_datetime + pd.Timedelta(days=1)).strftime("%d/%m/%Y")

        try:
            data = investpy.get_stock_historical_data(
                stock=ticker,
                country="United States",
                from_date=start_date,
                to_date=end_date
            )
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to fetch data for {ticker}: {e}")

        if data.empty:
            raise ValueError(f"[ERROR] No data returned for {ticker} up to {end_datetime}")

        # Standardize column names like yfinance
        data.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        }, inplace=True)

        # Add 'Adj Close' same as Close (Investpy doesn't provide it)
        data['Adj Close'] = data['Close']

        # Reorder columns like yf
        data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

        return data

    def get_stock(self, ticker: str, end_datetime: datetime) -> pd.DataFrame:
        """
        Returns historical stock data for a given ticker up to a specified end date.

        If a cached CSV file is found and contains sufficient data (i.e., up to `end_datetime`),
        it is loaded and returned. Otherwise, the function downloads the data using yfinance,
        saves it locally, and returns it.

        Args:
            ticker (str): The stock ticker symbol (e.g., 'AAPL', 'GOOGL').
            end_datetime (datetime): The end date for the desired historical data.

        Raises:
            ValueError: If the downloaded data is empty or invalid.

        Returns:
            pd.DataFrame: A DataFrame containing the stock data with dates up to `end_datetime`.
        """

        ticker = ticker.upper()
        date_str = end_datetime.strftime("%Y-%m-%d")
        file_path = os.path.join(self.data_dir, f"{ticker}_{date_str}.csv")

        # Return the data if it already exists
        pattern = re.compile(rf"{ticker}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv")
        for f in os.listdir(self.data_dir):
            match = pattern.match(f)
            if match:
                current_end_datetime =  datetime.strptime(f[len(ticker)+1: len(ticker)+11], '%Y-%m-%d')
                if end_datetime <= current_end_datetime:
                    data = pd.read_csv(os.path.join(self.data_dir,f), parse_dates=["Date"])
                    print(f"[INFO] {ticker} data found")
                    return data[data["Date"] <= end_datetime]
                     

        # Remove old versions of the same stock with outdated date
        pattern = re.compile(rf"{ticker}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv")
        for f in os.listdir(self.data_dir):
            match = pattern.match(f)
            if match:
                os.remove(os.path.join(self.data_dir, f))

        # Download from Yahoo Finance
        print(f"[INFO] Downloading {ticker} data until {end_datetime}")
        start_date = end_datetime - pd.Timedelta(days=5_000)
        data = yf.download(ticker, start=start_date, end=end_datetime + pd.Timedelta(days=1), interval="1d")

        # Raise Error if the data is empty 
        if data.empty:
            raise ValueError(f"[ERROR] No data returned for {ticker} up to {end_datetime}")

        # Drop a level of column (Yahoo Finance now add a new level of column)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Saving the data
        data.reset_index(inplace=True)
        data.to_csv(file_path, index=False)
        return data[data["Date"] <= end_datetime]

    def get_features(self, ticker: str, end_datetime: datetime, lookback: int) -> pd.DataFrame:
        """
        Returns feature-engineered data for a given stock ticker up to a specific date.

        If cached feature data exists and covers the `end_datetime`, it is loaded.
        Otherwise, features are computed from the historical stock data and saved locally.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL').
            end_datetime (datetime): The end date for the features to include.
            lookback (int): The number of past days to use for computing features.

        Raises:
            ValueError: If feature generation fails (i.e., resulting DataFrame is empty).

        Returns:
            pd.DataFrame: A DataFrame of computed features with dates up to `end_datetime`.
            pd.DataFrame: The raw stock data used for feature computation.
        """

        # Getting stock 
        stock = self.get_stock(ticker, end_datetime)

        ticker = ticker.upper()
        date_str = end_datetime.strftime("%Y-%m-%d")
        file_path = os.path.join(self.features_dir, f"{ticker}_{date_str}.csv")

        # Return the features if it already exists
        pattern = re.compile(rf"{ticker}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv")
        for f in os.listdir(self.features_dir):
            match = pattern.match(f)
            if match:
                current_end_datetime =  datetime.strptime(f[len(ticker)+1: len(ticker)+11], '%Y-%m-%d') 
                if end_datetime <= current_end_datetime:
                    data = pd.read_csv(os.path.join(self.features_dir,f), parse_dates=["date"])
                    return data[data["date"] <= end_datetime], stock

       
        # Remove old versions of the same stock with outdated date
        pattern = re.compile(rf"{ticker}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv")
        for f in os.listdir(self.features_dir):
            match = pattern.match(f)
            if match:
                os.remove(os.path.join(self.features_dir, f))

        print(f"[INFO] Computing features for {ticker} data until {end_datetime}")

        # Create the features from the stock and lookback
        df_features = create_features(stock, lookback)

        # Raise an error if the creation of the features dos not work
        if df_features.empty:
            raise ValueError(f"No features for {ticker} up to {end_datetime}")
        
        # Save the features data and return it with the stock
        df_features.to_csv(file_path, index=False)
        return df_features[df_features["date"] <= end_datetime], stock

    def train(self, ticker: str, model_name: str, end_datetime: datetime, lookback: int =20):
        """
        Trains a machine learning model on stock features up to a given date.

        This method retrieves (or generates) feature-engineered stock data,
        splits it into training and testing sets, normalizes the inputs, and
        trains the specified model.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL').
            model_name (str): Name of the model to train.
            end_datetime (datetime): The last date of data to use for training.
            lookback (int, optional): Number of past days to consider for feature generation. Defaults to 20.

        Returns:
            model: The trained model instance.
            StandardScaler: The fitted scaler used to normalize the data.
        """
        # Get features
        df_features, _ = self.get_features(ticker, end_datetime, lookback)

        # use 80% of data for training
        n = len(df_features)
        pourc = int(n*.8)

        data_train = df_features.iloc[:pourc]
        data_test = df_features.iloc[pourc:]

        x_train = data_train.drop(columns=["target","date"]).values
        x_test = data_test.iloc[:-1].drop(columns=["target","date"]).values

        y_train = data_train["target"].values
        y_test = data_test.iloc[:-1]["target"].values

        # Normalization
        x_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x_train)
        x_test = x_scaler.transform(x_test)

        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()  

            
        # Model Selection
        if model_name == "MLP":
            model = MLP_regression()
        elif model_name == "LSTM":
            model = RNN_regression()
        elif model_name == "XGBoost":
            model = XGBoost_regression()
        else :
            model = MLP_regression()

        model.fit(x_train, y_train) 

        # model info
        rmse = model.score(x_test, y_test, y_scaler)
        nb_trainable_parameters = model.get_nb_trainable_parameters()
        mae = model.get_mae(x_test, y_test, y_scaler)
        r2 = model.get_r2(x_test, y_test, y_scaler)
        model_name = model.__class__.__name__
        training_time = model.get_training_time()  
        features_used = len(x_test[0])  

        model_info = pd.DataFrame([{
            "model": model_name,
            # "rmse ($)": f"{rmse:.3f} $",
            "mae ($)": f"{mae:.3f} $",
            "r2 (%)": f"{r2 * 100:.2f} %",
            "nb_trainable_parameters": nb_trainable_parameters,
            "training_time (s)": f"{training_time:.2f} s",
            "features_used": features_used
        }])

        return model, x_scaler, y_scaler, model_info
    
    def get_prediction(self, ticker: str, model_name: str, end_datetime: datetime, days_to_predict: int = 10, lookback: int = 20) -> pd.DataFrame:
        """
        Generates past and future stock price predictions using a trained model.

        This method loads an existing model if available, or trains one if not.
        It uses historical features to predict known data, and then simulates
        predictions for a number of future days based on the last known data.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL').
            model_name (str): Name of the model to use (e.g., 'MLP').
            end_datetime (datetime): Last date for which real data is used.
            days_to_predict (int, optional): Number of future days to predict. Defaults to 10.
            lookback (int, optional): Number of past days used for feature generation. Defaults to 20.

        Returns:
            pd.DataFrame: DataFrame including historical data, model predictions on it,
                        and extended predictions for future dates.
        """
        model_path = os.path.join(self.model_dir, f"{ticker}_{model_name}.pkl")
        
        if os.path.exists(model_path):
            print("[INFO] Model already trained ! ")
            with open(model_path, "rb") as file:
                model, x_scaler, y_scaler, model_info = pickle.load(file)
        else : 
            print("[INFO] Need to run the model")
            model, x_scaler, y_scaler, model_info = self.train(ticker, model_name, end_datetime, lookback)

            with open(model_path, "wb") as file:
                pickle.dump((model, x_scaler, y_scaler, model_info), file)
        
        df_features, stock = self.get_features(ticker, end_datetime, lookback)
        df_predicted = df_features.copy()

        prediction = model.predict(x_scaler.transform(df_features.drop(columns=["target", "date"]).values))
        df_predicted["predicted"] = y_scaler.inverse_transform(prediction.reshape(-1, 1)).ravel()

        stock_predicted = stock.copy()
        
        # Prediction of next days 

        for _ in range(days_to_predict):
            # Populate stock_predicted 
            last_line = df_predicted.iloc[-1]
            next_date = last_line["date"]
            next_close = last_line["predicted"]
            next_open = last_line["Close_(-1)"]
            next_high = max(next_close, next_open) 
            next_low = min(next_close, next_open) 
            next_volume = -1
            next_line = pd.DataFrame([{
                "Date": next_date,
                "Close": next_close,
                "High": next_high,
                "Low": next_low,
                "Open": next_open,
                "Volume": next_volume,
            }])
            stock_predicted = pd.concat([stock_predicted, next_line], ignore_index=True)

            stock_predicted = stock_predicted[-2*lookback:]
            next_features = create_features(stock_predicted, lookback)
            target_next_day = model.predict(x_scaler.transform(next_features.drop(columns=["target","date"])))
            next_features["predicted"]= y_scaler.inverse_transform(target_next_day.reshape(-1, 1)).ravel()
            # Populate df_predicted
            df_predicted = pd.concat([df_predicted, next_features], ignore_index=True)
    
        return {
            "df_predicted": df_predicted,
            "model_info": model_info,
        }

