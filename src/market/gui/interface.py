import streamlit as st
from datetime import datetime, timedelta
import os
from market.gui.GuiManager import GuiManager

def main():

    # ========== manager ==========
    gui_manager = GuiManager()


    # ========== CONFIG ==========

    st.set_page_config(
        page_title="Stock Prediction App",  
        page_icon="📈",  
        layout="wide",
    )

    # image = Image.open("path/to/your/logo.png") 
    # st.image(image, width=100) 
    #  
    st.title("Stock Prediction")
    st.markdown("""
    ### 📈 Welcome to the Stock Prediction App (by François Goybet)
    """)
    st.info("""
    Start by selecting the stock you want to **view** and **predict**.  
    Then, choose a **model** and specify the number of **days to forecast**.
    """)

    # ========== SECTION 1 : STOCK & DATA VISUALIZATION ==========
    st.header("1. Select Stock and View Historical Data")

    # Dropdown to select stock
    stock_choice = st.selectbox("Choose a stock", gui_manager.stock_name)
    stock_choice = gui_manager.stock_tickers[gui_manager.stock_name.index(stock_choice)]

    # Extra info for users
    st.info("""
    This interactive chart displays a **candlestick view** of the selected stock 📊  
    Use the **zoom** and **pan** tools, or slide through the timeline to explore price movements in detail.
    """)

    # Placeholder for future plot
    graph_placeholder = st.empty()
    end_datetime = datetime.now().replace(hour=0, second=0, minute=0, microsecond=0) + timedelta(days=1)

    fig = gui_manager.plot_ticker(stock_choice, end_datetime)
    graph_placeholder.plotly_chart(fig, use_container_width=True)

    # ========== SECTION 2 : PREDICTION SETTINGS ==========
    st.header("2. Prediction")

    model_full_name_dict = {
        "MultiLayer Perceptron (MLP)": "The **MultiLayer Perceptron (MLP)** is a feedforward artificial neural network that learns complex nonlinear relationships between features, making it suitable for regression tasks based on structured input data.",
        "Long short-term memory (LSTM)": "The **Long Short-Term Memory (LSTM)** is a type of recurrent neural network designed to capture temporal dependencies in sequential data, making it well-suited for time series prediction like stock prices.",
        # "XGBoost": "The **XGBoost** model is an optimized gradient boosting algorithm that performs exceptionally well for both classification and regression tasks. It is particularly powerful for handling structured/tabular data and can capture complex relationships between features efficiently."
    }

    model_short_name_dict = {
        "MultiLayer Perceptron (MLP)": "MLP",  
        "Long short-term memory (LSTM)": "LSTM",   
        # "XGBoost": "XGBoost"
    }

    # Model selection with full name options
    st.markdown("### Select a Prediction Model")
    model_choice_full = st.selectbox("Choose prediction model", model_short_name_dict.keys())

    # Show description of the selected model
    st.info(model_full_name_dict.get(model_choice_full, "No description available for the selected model. It consists of multiple layers of neurons that can model complex relationships in the data. "))

    # Days to predict
    st.markdown("### Set Prediction Horizon")
    days_to_predict = st.slider("How many days to predict?", min_value=1, max_value=30, value=1, step=1)

    # Info about prediction process and Close price
    st.markdown("""
    The model will forecast the **closing price** (market closing value) of the stock for the next **X days**. This is the final price at which the stock will settle at the end of each trading day.

    Click the **Run prediction** button below to start generating the forecast for the stock price.
    """)

    # Predict button
    if st.button("🔮 Run prediction"):
        with st.spinner("Generating prediction... Please wait..."):
            # Get the short name of the model for the function argument
            model_choice_short = model_short_name_dict.get(model_choice_full, "MLP")
            
            # Call the prediction function with the short model name
            fig = gui_manager.get_prediction(stock_choice, model_choice_short, end_datetime, days_to_predict)
            st.success("Prediction completed ✅")

            # Placeholder for prediction plot
            graph_placeholder = st.empty()
            graph_placeholder.plotly_chart(fig, use_container_width=True)
            st.info("More informations will come here ...")

