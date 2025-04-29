# Market Prediction App

This application predicts the closing price of selected stock tickers (e.g., AAPL, TSLA, MSFT, etc.) using historical data. You can interact with the app and see the predictions for different stocks. It is built using simple technical analysis methods such as moving averages, variance, and stock values over the past 20 days.

You can access the app here: [Market Prediction App](https://market-p8qwtca6ent3mcsqvkfrkc.streamlit.app/).

## Features
- **Stock Prediction**: Predict the closing price for stock tickers like AAPL, TSLA, MSFT using basic models.
- **Graphical Analysis**: The app analyzes and visualizes the stock data using moving averages and variance over the past 20 days.
- **Models**: Currently, the app uses two models:
  - A simple **MLP (Multi-layer Perceptron)** model.
  - A basic **LSTM (Long Short-Term Memory)** model.

## Current Limitations
- The app is still in development and far from finished.
- The interface needs improvement (e.g., displaying model scores).
- The models can be enhanced (e.g., implementing cross-validation).
- Feature engineering could be improved (e.g., exploring correlations between multiple stocks and integrating them).

## Future Improvements
- **Better Interface**: Improve the user interface and display the models' performance metrics.
- **Model Refinement**: Fine-tune the models and implement cross-validation to improve prediction accuracy.
- **Feature Engineering**: Explore and integrate additional features, such as combining multiple stocks' data and analyzing correlations between them.

## Technologies
- Streamlit (for app deployment)
- Numpy, Pandas (for data manipulation)
- TensorFlow/Keras (for MLP and LSTM models)
- Matplotlib, Plotly (for data visualization)

---

Feel free to explore the app, and your feedback is highly appreciated!
