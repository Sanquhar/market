from data.manager import DataManager
from visualization import visualization
from model import preprocessing
from model.model import RNN_regression, MLP_regression
import numpy as np
from sklearn.preprocessing import StandardScaler

class Routine_next_day:

    def __init__(self) -> None:

        self.sequence_length = 10
        self.input_dim = 4
        self.output_dim = 1

        ticker_symbols = [
            "TSLA", # Tesla
            "AAPL", # Apple
            "NVDA", # Nvidia
            "BTC-USD", # Bitcoin
        ]
        for ticker in ticker_symbols:

            self.apply(ticker=ticker)


    def apply(self, ticker : str):

        print(f"ROUTINE NEXT DAY for {ticker}")

        # data and dataset
        print("Querying raw data and computing dataset...")
        data = DataManager.get_stock(ticker)
        x,y,y_dates = preprocessing.create_dataset_with_dates(data)

        # save dataset
        print("Saving the dataset...")
        np.save("generated/"+ticker+"/x.npy",x)
        np.save("generated/"+ticker+"/y.npy",y)
        np.save("generated/"+ticker+"/y_dates.npy",y_dates)

        # load dataset
        print("Loading the dataset...")
        x = np.load("generated/"+ticker+"/x.npy")
        y = np.load("generated/"+ticker+"/y.npy")
        y_dates = np.load("generated/"+ticker+"/y_dates.npy",allow_pickle=True)

        # reshape 
        print("Reshaping, creating train and test set, and normalizing the dataset...")
        n = len(x)
        x = x.reshape((n,self.sequence_length,self.input_dim))
        y = y.reshape((n,1,self.output_dim))

        # compute train and test
        pourc = int(n*.8)
        
        x_train = x[:pourc]
        x_test = x[pourc:]
        
        y_train = y[:pourc]
        y_test = y[pourc:]

        y_dates_train = y_dates[:pourc]
        y_dates_test = y_dates[pourc:]

        # normalization
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
        x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

        # model and save model
        print("Fitting the model...")
        # regressor = RNN_regression(x.shape[1],x.shape[2],y.shape[2])
        regressor = MLP_regression(x.shape[1],x.shape[2],y.shape[2])
        regressor.fit(x_train,y_train)
        regressor.save(path = "generated/"+ticker+"/model.keras")

        # prediction
        print("Prediction for test set...")
        y_pred = regressor.predict(x_test)
        data["Close_predict"] = None
        data.loc[y_dates_test,"Close_predict"] = y_pred

        # prediction tomorrow
        print("Prediction for tomorrow...")
        X, last_date = preprocessing.create_dataset_one_predict(data)
        X = X.reshape((1,self.sequence_length,self.input_dim))
        X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        # visualization
        print("Saving graphs...")
        fig = visualization.scatter_price(data, ticker)
        path = f"generated/figures/scatter_price_{ticker}.html"
        fig.write_html(path)

        print(f"Routine {ticker} DONE ! See graph at {path}.",end="\n\n")

class Routine_several_days:

    def __init__(self) -> None:

        self.sequence_length = 10
        self.input_dim = 4
        self.output_dim = 1
        self.sequence_length_output = 5

        # ticker_symbols = [
        #     "TSLA", # Tesla
        #     "AAPL", # Apple
        #     "NVDA", # Nvidia
        #     "BTC-USD", # Bitcoin
        # ]
        # for ticker in ticker_symbols:

        ticker = "AAPL"

        # self.apply(ticker=ticker)


    def apply(self, ticker : str):

        print(f"ROUTINE SEVERAL DAYS for {ticker}")

        # data and dataset
        print("Querying raw data and computing dataset...")
        data = DataManager.get_stock(ticker)
        x,y,y_dates = preprocessing.create_dataset_with_dates(data)

        # save dataset
        print("Saving the dataset...")
        np.save("generated/"+ticker+"/x.npy",x)
        np.save("generated/"+ticker+"/y.npy",y)
        np.save("generated/"+ticker+"/y_dates.npy",y_dates)

        # load dataset
        print("Loading the dataset...")
        x = np.load("generated/"+ticker+"/x.npy")
        y = np.load("generated/"+ticker+"/y.npy")
        y_dates = np.load("generated/"+ticker+"/y_dates.npy",allow_pickle=True)

        # reshape 
        print("Reshaping, creating train and test set, and normalizing the dataset...")
        n = len(x)
        x = x.reshape((n,self.sequence_length,self.input_dim))
        y = y.reshape((n,1,self.output_dim))

        # compute train and test
        pourc = int(n*.8)
        
        x_train = x[:pourc]
        x_test = x[pourc:]
        
        y_train = y[:pourc]
        y_test = y[pourc:]

        y_dates_train = y_dates[:pourc]
        y_dates_test = y_dates[pourc:]

        # normalization
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
        x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

        # model and save model
        print("Fitting the model...")
        # regressor = RNN_regression(x.shape[1],x.shape[2],y.shape[2])
        regressor = MLP_regression(x.shape[1],x.shape[2],y.shape[2])
        regressor.fit(x_train,y_train)
        regressor.save(path = "generated/"+ticker+"/model.keras")

        # prediction
        print("Prediction for test set...")
        y_pred = regressor.predict(x_test)
        data["Close_predict"] = None
        data.loc[y_dates_test,"Close_predict"] = y_pred

        # prediction tomorrow
        print("Prediction for tomorrow...")
        X, last_date = preprocessing.create_dataset_one_predict(data)
        X = X.reshape((1,self.sequence_length,self.input_dim))
        X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        # visualization
        print("Saving graphs...")
        fig = visualization.scatter_price(data, ticker)
        path = f"generated/figures/scatter_price_{ticker}.html"
        fig.write_html(path)

        print(f"Routine {ticker} DONE ! See graph at {path}.",end="\n\n")