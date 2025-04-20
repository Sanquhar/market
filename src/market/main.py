from data.DataManager import DataManager
from model import preprocessing
import pandas as pd
# from routines.routine import Routine_next_day, Routine_several_days
from model.model import MLP_regression
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# from visualization import visualization
from datetime import datetime
import os

def main() -> None:
    """
    main function
    """

    data_manager = DataManager()
    lookback = 20
    df_features = data_manager.get_prediction("AAPL", "MLP", datetime(2025, 4, 19), 20)
    print(df_features)
    # print(stock)
    # df_predicted = df_features.copy()

    # df_predicted["predicted"] = model.predict(scaler.transform(df_features.drop(columns=["target", "date"]).values))
    
    # stock_predicted = stock.copy()
    # ##### Next_days ####
    
    # for _ in range(10):
    #     last_line = df_predicted.iloc[-1]
    #     next_date = last_line["date"]
    #     next_close = last_line["predicted"]
    #     next_open = last_line["Close_(-1)"]
    #     next_high = max(next_close, next_open) 
    #     next_low = min(next_close, next_open) 
    #     next_volume = -1
    #     next_line = pd.DataFrame([{
    #         "Date": next_date,
    #         "Close": next_close,
    #         "High": next_high,
    #         "Low": next_low,
    #         "Open": next_open,
    #         "Volume": next_volume,
    #     }])
    #     stock_predicted = pd.concat([stock_predicted, next_line], ignore_index=True)

    #     stock_predicted = stock_predicted[-2*lookback:]
    #     next_features = preprocessing.create_features(stock_predicted, lookback)
    #     target_next_day = model.predict(scaler.transform(next_features.drop(columns=["target","date"])))
    #     next_features["predicted"]= target_next_day
       
    #     df_predicted = pd.concat([df_predicted, next_features], ignore_index=True)
   



if __name__ == '__main__':
    main()