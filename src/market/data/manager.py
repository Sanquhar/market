import pandas as pd
import yfinance as yf
import numpy as np

class DataManager:

    def __init__(self, path : str) -> None:
        self.df_AAPL = pd.read_csv(path + "AAPL.csv")
        self.df_AAPL['Date'] = pd.to_datetime(self.df_AAPL['Date'])
        self.df_TSLA = pd.read_csv(path + "TSLA.csv")
        self.df_TSLA['Date'] = pd.to_datetime(self.df_TSLA['Date'])
        # print(self.df_AAPL.columns)
        # print(self.df_AAPL.info())
        # print(self.df_AAPL.describe())
        # print(self.df_AAPL['Date'].unique())
    
    def get_stock(ticker_symbol : str, period : str = "max") -> pd.DataFrame:
        ticker = yf.Ticker(ticker_symbol)
        historical_data = ticker.history(period=period)
        return historical_data

