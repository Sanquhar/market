import os
from market.data.DataManager import DataManager
import plotly.graph_objects as go
import datetime as datetime

class GuiManager:
    """
    Class responsible for managing the graphical interface and generating
    plots for stock data and predictions.
    """
    def __init__(self):
        # List of supported stock tickers
        self.stock_tickers = ["AAPL", "TSLA", "MSFT", "AIR.PA", "VIE.PA"]
        self.stock_name = ["Apple", "Tesla", "Microsoft", "Airbus", "L'Oréal"]

        # Instance of DataManager to handle data operations
        self.data_manager = DataManager()

    def plot_ticker(self, ticker: str, end_datetime: datetime,  days_shown: int = 300) -> go.Figure:
        """
        Generates a candlestick chart for a given ticker over a recent time window.

        Args:
            ticker (str): Stock ticker symbol (must be in predefined list).
            end_datetime (datetime): Last date to display on the chart.
            days_shown (int): Number of days before end_datetime to include. Default is 300.

        Returns:
            go.Figure: A Plotly candlestick chart of the stock prices.
        """
        
        # Validate that the ticker is supported
        if ticker not in self.stock_tickers:
            raise ValueError(f"{ticker} is not in {self.stock_tickers}")

        # Load historical stock data up to the given date
        df = self.data_manager.get_stock(ticker, end_datetime)

        # Filter to keep only the last `days_shown` days
        df = df[df["Date"] >= end_datetime - datetime.timedelta(days=days_shown)]
        
        # Create candlestick chart
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
        """
        Retrieves and plots model predictions for a given stock ticker.

        Args:
            ticker (str): Stock ticker symbol.
            model_name (str): Name of the model to use for predictions.
            end_datetime (datetime): Last known date for real data.
            days_to_predict (int): Number of days to forecast into the future.
            lookback (int): Number of past days used to build features. Default is 20.
            days_shown (int): Number of days to display in the plot. Default is 300.

        Returns:
            go.Figure: A Plotly line chart showing predicted vs actual values.
        """

        df_predicted = self.data_manager.get_prediction(ticker, model_name, end_datetime, days_to_predict)
        fig = go.Figure(data= [
            go.Scatter(arg={
                'x':df_predicted.iloc[-days_shown:]["date"], 
                'y':df_predicted.iloc[-days_shown:]["target"],
                # 'c':'blue
                'name': 'Actual Close',
                'mode': 'lines+markers',
            }),
            # Retrieve the full DataFrame with past + future predictions
            go.Scatter(arg={
                'x':df_predicted.iloc[-days_shown:]["date"], 
                'y':df_predicted.iloc[-days_shown:]["predicted"],
                # 'c':'red',
                'name': 'Predicted Close',
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