import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

def create_features(df, lookback=20):
    """
    This function generates features for predicting the stock's closing price 
    from historical stock data. It calculates technical indicators such as 
    returns, moving averages, and volatility, and then formats the data for 
    supervised learning.
    
    Args:
        df (pd.DataFrame): A DataFrame containing the stock's historical data. 
                           It must have the following columns:
                           - 'Date': the date of the stock data (datetime type)
                           - 'Close': the stock's closing price
                           - 'High': the highest price of the stock during the day
                           - 'Low': the lowest price of the stock during the day
                           - 'Open': the stock's opening price
                           - 'Volume': the volume of stocks traded
        lookback (int): The number of previous days to consider when creating the features. 
                        Default is 20 days.

    Returns:
        pd.DataFrame: A DataFrame with generated features and the target for each date.
                      The columns include information like returns, moving averages,
                      volatility, and the target (closing price to predict).
    """
    
    # Create a copy of the DataFrame to avoid modifying the original data
    df = df.copy()
    
    # Calculate daily returns
    df['return'] = df['Close'].pct_change()

    # Calculate moving averages over 5, 10, and 20 days
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    
    # Calculate volatility over 5 and 20 days
    df['volatility_5'] = df['Close'].rolling(window=5).std()
    df['volatility_20'] = df['Close'].rolling(window=20).std()
    
    # Drop rows with missing values (which result from the rolling calculations)
    df.dropna(inplace=True)
    
    # Reset the index to avoid issues after dropping rows
    df.reset_index(drop=True, inplace=True)

    # List of features to include for each lookback window
    features = ['Close', 'High', 'Low', 'Open', 'return', 
                'ma_5', 'ma_10', 'ma_20', 'volatility_5', 'volatility_20']
    
    # List to store the formatted data for supervised learning
    data = []
    
    # Loop to create features for each lookback window
    for i in tqdm(range(lookback, len(df)-1)):
        # Select the features for the 'lookback' previous days
        feature_row = df[features].iloc[i+1-lookback:i+1].values.flatten()
        
        # Target: the closing price of the next day
        target = df['Close'].iloc[i+1]
        
        # Date corresponding to the next day
        date = df['Date'].iloc[i+1]
        
        # Append the features and target to the data list
        data.append(np.concatenate([feature_row, [target, date]]))
    
    # Create the row with the features of the last 'lookback' days and the next date
    last_features = df[features].iloc[-lookback:].values.flatten()
    next_date = df['Date'].iloc[-1]
    
    # If the last date is a Friday (or later), predict for the following Monday
    if next_date.weekday() >= 4:
        next_date += timedelta(days=7 - next_date.weekday())
    else:
        next_date += timedelta(days=1)
    
    # The target for the last day is unknown, so it is set to None
    next_target = None
    
    # Append this last row to the data list
    data.append(np.concatenate([last_features, [next_target, next_date]]))
    
    # Generate column names for the final DataFrame
    columns = [feature + f"_({i-lookback-1})" for i in range(1, lookback+1) for feature in features] + ['target', 'date']
    
    # Create the final DataFrame with the features and target values
    df_features = pd.DataFrame(data, columns=columns)

    return df_features
