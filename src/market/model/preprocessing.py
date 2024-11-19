import numpy as np
from datetime import datetime
import pandas as pd

def create_dataset_with_dates(df, sequence_length=10, sequence_length_output=1):
    X, y, y_dates = [], [], []
    position_variables_x = np.where(df.columns.isin(["Open","High","Low","Close"]))[0]
    position_variables_y = np.where(df.columns.isin(["Close"]))[0]

    for i in range(len(df) - sequence_length - sequence_length_output + 1):
        print("{:.2f}% performed".format(100*(i+1) / (len(df) - sequence_length - sequence_length_output + 1)),end="\r")
        X.append(df.iloc[i:i + sequence_length, position_variables_x].values)
        y.append(df.iloc[i + sequence_length:i+sequence_length+sequence_length_output, position_variables_y].values)  
        y_dates.append(df.index[i + sequence_length])  

    print()
    return np.array(X), np.array(y), np.array(y_dates)

def create_dataset_one_predict(df, sequence_length=10, last_date = None):

    position_variables_x = np.where(df.columns.isin(["Open","High","Low","Close"]))[0]
    position_variables_y = np.where(df.columns.isin(["Close"]))[0]
    X = df.iloc[-sequence_length:,position_variables_x]
    last_date = df.index[-1]

    return np.array(X), last_date

from datetime import datetime, timedelta

def get_next_dates(start_date, n):
    days = []
    count = 0
    current_day = start_date
    
    while count < n:
        current_day += timedelta(days=1)
        
        if current_day.weekday() < 5:  
            days.append(current_day)
            count += 1
    
    return days

def find_closest_date(df, date_str):
    target_date = pd.to_datetime(date_str)
    target_date = target_date.tz_localize('UTC').tz_convert('America/New_York')
    diff = abs((df.index - target_date))
    return df.index[np.argmin(diff)]
