import streamlit as st
from datetime import datetime, date, timedelta
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
end_datetime = datetime.now().replace(hour=0, second=0, minute=0, microsecond=0) + timedelta(days=1)

fig = gui_manager.plot_ticker(stock_choice, end_datetime)
graph_placeholder.plotly_chart(fig, use_container_width=True)

# ========== SECTION 2 : PREDICTION SETTINGS ==========
st.header("2. Prediction Settings")

# Model selection
model_choice = st.selectbox("Choose prediction model", ["MLP"])

days_to_predict = st.slider("How many days to predict ?" , min_value=1, max_value=30, value=1, step=1)

# Predict button
if st.button("🔮 Run prediction"):
    with st.spinner("Generating prediction..."):
        fig = gui_manager.get_prediction(stock_choice, model_choice, end_datetime, days_to_predict)
        st.success("Prediction completed ✅")
        graph_placeholder = st.empty()
        graph_placeholder.plotly_chart(fig, use_container_width=True)