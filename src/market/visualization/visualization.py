import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def scatter_price(data : pd.DataFrame, ticker : str) -> go.Figure:
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='markers',
        name='Close',
        marker=dict(color='blue',size=5)
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close_predict'],
        mode='markers',
        name='Close_predict',
        marker=dict(color='red',size=5)
    ))

    fig.update_layout(
        title= f"Prediction Close value for {ticker}" 
    )

    return fig
