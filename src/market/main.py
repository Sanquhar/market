from data.manager import DataManager
from model import preprocessing
from routines.routine import Routine_next_day
from model.model import MLP_regression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from visualization import visualization

def main() -> None:
    """
    main function
    """

    # routine1 = Routine_next_day()

    sequence_length = 30
    input_dim = 4
    output_dim = 1
    sequence_length_output = 10

    ticker = "AAPL"

    print(f"ROUTINE SEVERAL DAYS for {ticker}")

    # data and dataset
    print("Querying raw data and computing dataset...")
    data = DataManager.get_stock(ticker)
    x, y, y_dates = preprocessing.create_dataset_with_dates(
        data,
        sequence_length=sequence_length,
        sequence_length_output=sequence_length_output
        )

    # reshape 
    print("Reshaping, creating train and test set, and normalizing the dataset...")
    n = len(x)
    x = x.reshape((n,sequence_length,input_dim))
    y = y.reshape((n,sequence_length_output,output_dim))

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

    # score the model with test set
    print("Scoring the model...")
    print("Score : ", regressor.score(x_test,y_test))

    # prediction next days
    print("Prediction for next days...")
    X, last_date = preprocessing.create_dataset_one_predict(data,sequence_length=sequence_length)
    X = X.reshape((1,sequence_length,input_dim))
    X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    predicted_values = regressor.predict(X)
    print(last_date)
    print(regressor.predict(X))

    # create new dataframe
    last_date = preprocessing.find_closest_date(data, "2024-09-10")

    dates = preprocessing.get_next_dates(last_date, sequence_length_output)
    next_dates = [date for date in dates if date not in data.index]
    print(dates)
    new_data = pd.DataFrame(index=next_dates, columns=data.columns)
    print(data.index.values[-10:])
    data = pd.concat([data, new_data])

    data.loc[dates, 'Close_predict'] = predicted_values.ravel()
    print(len(dates)) 
    print(dates)
    print(predicted_values.shape)
    print(data.index.values[-15:])
    fig = visualization.scatter_price(data=data,ticker=ticker)
    fig.show()

    print(data.index)

if __name__ == '__main__':
    main()