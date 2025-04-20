import os
from market.data.DataManager import DataManager
import plotly.graph_objects as go
import datetime as datetime

class GuiManager:

    def __init__(self):
        self.stock_tickers = ["AAPL", "TSLA", "MSFT"]
        self.data_manager = DataManager()

    def plot_ticker(self, ticker: str, end_datetime: datetime,  days_shown: int = 300) -> go.Figure:
        if ticker not in self.stock_tickers:
            raise ValueError(f"{ticker} is not in {self.stock_tickers}")

        df = self.data_manager.get_stock(ticker, end_datetime)

        df = df[df["Date"] >= end_datetime - datetime.timedelta(days=days_shown)]
        
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

    def get_prediction(self, ticker: str, model_name: str, end_datetime: datetime, days_to_predict: int, lookback: int = 20, days_shown: int = 300):
        df_predicted = self.data_manager.get_prediction(ticker, model_name, end_datetime, days_to_predict)
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