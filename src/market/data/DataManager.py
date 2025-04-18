import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import re
from market.model.model import MLP_regression
from market.model.preprocessing import create_features
from sklearn.preprocessing import StandardScaler
import pickle

class DataManager:

    def __init__(
            self, 
            data_dir="data/stocks", 
            model_dir="generated/model",
            features_dir="generated/features"
            ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.features_dir = features_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

    def get_stock(self, ticker: str, end_datetime: datetime, precision: str = 'hour') -> pd.DataFrame:

        if precision == 'hour':
            end_datetime = end_datetime.replace(second=0, microsecond=0, minute=0)

        ticker = ticker.upper()
        date_str = end_datetime.strftime("%Y-%m-%d")
        file_path = os.path.join(self.data_dir, f"{ticker}_{date_str}.csv")

        pattern = re.compile(rf"{ticker}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv")
        for f in os.listdir(self.data_dir):
            match = pattern.match(f)
            if match:
                current_end_datetime =  datetime.strptime(f[len(ticker)+1: len(ticker)+11], '%Y-%m-%d') 
                if end_datetime <= current_end_datetime:
                    data = pd.read_csv(os.path.join(self.data_dir,f), parse_dates=["Date"])
                    return data[data["Date"] <= end_datetime]
                     

        # Remove old versions of the same stock with outdated date
        pattern = re.compile(rf"{ticker}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv")
        for f in os.listdir(self.data_dir):
            match = pattern.match(f)
            if match:
                # pass
                os.remove(os.path.join(self.data_dir, f))

        print(f"Downloading {ticker} data until {end_datetime}")
        start_date = end_datetime - pd.Timedelta(days=5_000)
        data = yf.download(ticker, start=start_date, end=end_datetime + pd.Timedelta(days=1), interval="1d")

        if data.empty:
            raise ValueError(f"No data returned for {ticker} up to {end_datetime}")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        data.reset_index(inplace=True)
        data.to_csv(file_path, index=False)
        return data[data["Date"] <= end_datetime]

    def get_features(self, ticker: str, end_datetime: datetime, lookback: int, precision: str = 'hour') -> pd.DataFrame:
        
        if precision == 'hour':
            end_datetime = end_datetime.replace(second=0, microsecond=0, minute=0)

        ticker = ticker.upper()
        date_str = end_datetime.strftime("%Y-%m-%d")
        file_path = os.path.join(self.features_dir, f"{ticker}_{date_str}.csv")

        pattern = re.compile(rf"{ticker}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv")
        for f in os.listdir(self.features_dir):
            match = pattern.match(f)
            if match:
                current_end_datetime =  datetime.strptime(f[len(ticker)+1: len(ticker)+11], '%Y-%m-%d') 
                if end_datetime <= current_end_datetime:
                    data = pd.read_csv(os.path.join(self.features_dir,f), parse_dates=["date"])
                    return data[data["date"] <= end_datetime]

        # if os.path.exists(file_path):

        #     return pd.read_csv(file_path, parse_dates=["date"])

        # Remove old versions of the same stock with outdated date
        pattern = re.compile(rf"{ticker}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv")
        for f in os.listdir(self.features_dir):
            match = pattern.match(f)
            if match:
                # pass
                os.remove(os.path.join(self.features_dir, f))

        print(f"Computing features for {ticker} data until {end_datetime}")

        stock = self.get_stock(ticker, end_datetime)
        df_features = create_features(stock, lookback)

        if df_features.empty:
            raise ValueError(f"No features for {ticker} up to {end_datetime}")
        
        df_features.to_csv(file_path, index=False)
        return df_features[df_features["date"] <= end_datetime]


    def train(self, ticker: str, model_name: str, end_datetime: datetime, lookback: int =20):

        df_features = self.get_features(ticker, end_datetime, lookback)

        n = len(df_features)
        pourc = int(n*.8)

        data_train = df_features.iloc[:pourc]
        data_test = df_features.iloc[pourc:]

        x_train = data_train.drop(columns=["target","date"]).values
        x_test = data_test.iloc[:-1].drop(columns=["target","date"]).values

        y_train = data_train["target"].values
        y_test = data_test.iloc[:-1]["target"].values

        # normalization
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        if model_name == "MLP":
            model = MLP_regression()
        else :
            model = MLP_regression()

        model.fit(x_train, y_train) 

        return model, scaler
    
    def get_prediction(self, ticker: str, model_name: str, end_datetime: datetime, lookback: int = 20) -> pd.DataFrame:
        
        model_path = os.path.join(self.model_dir, f"{ticker}_{model_name}.pkl")
        
        if os.path.exists(model_path):
            print("Model already trained ! ")
            with open(model_path, "rb") as file:
                model, scaler = pickle.load(file)
        else : 
            print("Need to run the model")
            model, scaler = self.train(ticker, model_name, end_datetime, lookback)

            with open(model_path, "wb") as file:
                pickle.dump((model, scaler), file)
        
        df_features = self.get_features(ticker, end_datetime, lookback)
        df_predicted = df_features.copy()
        df_predicted["predicted"] = model.predict(scaler.transform(df_features.drop(columns=["target", "date"]).values))
        
        return df_predicted

