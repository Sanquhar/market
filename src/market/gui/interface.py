import streamlit as st
from datetime import datetime
import os
import sys
import os
from GuiManager import GuiManager

# ========== manager ==========
gui_manager = GuiManager()


# ========== CONFIG ==========
st.set_page_config(page_title="Stock Prediction", layout="centered")

st.title("Stock Prediction")
st.markdown("Welcome Back")

# ========== SECTION 1 : STOCK & DATA VISUALIZATION ==========
st.header("1. Select Stock and View Historical Data")

# Dropdown to select stock
stock_choice = st.selectbox("Choose a stock", gui_manager.stock_tickers)

# Placeholder for future plot
st.info(f"📊 Historical data for **{stock_choice}** will be displayed here.")
graph_placeholder = st.empty()
fig = gui_manager.plot_ticker(stock_choice)
graph_placeholder.plotly_chart(fig, use_container_width=True)

# ========== SECTION 2 : DATA ANALYSIS ==========
st.header("2. Data Analysis")

# ========== SECTION 3 : PREDICTION SETTINGS ==========
st.header("3. Prediction Settings")

# Model selection
model_choice = st.selectbox("Choose prediction model", ["MLP"])

# Start date for prediction
end_datetime = st.date_input("End Datetime", datetime.today())
end_datetime = datetime(end_datetime.year, end_datetime.month, end_datetime.day)

# Predict button
if st.button("🔮 Run prediction"):
    with st.spinner("Generating prediction..."):
        fig = gui_manager.get_prediction(stock_choice, model_choice, end_datetime)
        st.success("Prediction completed ✅")
        graph_placeholder = st.empty()
        graph_placeholder.plotly_chart(fig, use_container_width=True)