import os
from market.data.DataManager import DataManager
import plotly.graph_objects as go
import datetime as datetime

class GuiManager:

    def __init__(self):
        self.stock_tickers = ["AAPL", "TSLA", "MSFT"]
        self.data_manager = DataManager()
        self.stocks = dict()
        for ticker in self.stock_tickers:
            self.stocks[ticker] = self.data_manager.get_stock(ticker=ticker, end_datetime=datetime.datetime(2025, 4, 12))
        
    def plot_ticker(self, ticker: str, days_shown: int = 300) -> go.Figure:
        if ticker not in self.stock_tickers:
            raise ValueError(f"{ticker} is not in {self.stock_tickers}")

        df = self.stocks[ticker].copy()

        df = df[df["Date"] >= datetime.datetime.now() - datetime.timedelta(days=days_shown)]
        
        fig = go.Figure(data=[go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])

        fig.update_layout(
            title=f"{ticker} - Last {days_shown} Days",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark"
        )

        return fig

    def get_prediction(self, ticker: str, model_name: str, end_datetime: datetime, lookback: int = 20, days_shown: int = 300):
        df_predicted = self.data_manager.get_prediction(ticker, model_name, end_datetime)
        fig = go.Figure(data= [
            go.Scatter(arg={
                'x':df_predicted.iloc[-days_shown:]["date"], 
                'y':df_predicted.iloc[-days_shown:]["target"],
                # 'c':'blue
                'name': 'target',
                'mode': 'lines+markers',
            }),
            go.Scatter(arg={
                'x':df_predicted.iloc[-days_shown:]["date"], 
                'y':df_predicted.iloc[-days_shown:]["predicted"],
                # 'c':'red',
                'name': 'predicted',
                'mode': 'lines+markers',
            }),       
        ])
        fig.update_layout(
            title=f"{ticker} Prediction - Last {days_shown} Days",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark"
        )
        return fig 