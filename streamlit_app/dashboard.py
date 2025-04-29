import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath("src"))
from preprocessing import load_cmapss_dataset

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.title("🛠️ Predictive Maintenance - Enhanced Dashboard")

st.markdown("Predict Remaining Useful Life (RUL) with sensor data and monitor trends in real time.")

# Sidebar: User Input
st.sidebar.header("🔧 Input Sensor Data")
op_setting_1 = st.sidebar.number_input("Operational Setting 1", value=0.0)
op_setting_2 = st.sidebar.number_input("Operational Setting 2", value=0.0)
op_setting_3 = st.sidebar.number_input("Operational Setting 3", value=0.0)

sensor_inputs = {}
for i in range(1, 22):
    sensor_inputs[f'sensor_measurement_{i}'] = st.sidebar.number_input(f"Sensor {i}", value=np.random.uniform(0.0, 1.0))

# Prediction
payload = {
    "op_setting_1": op_setting_1,
    "op_setting_2": op_setting_2,
    "op_setting_3": op_setting_3,
    **sensor_inputs
}

predict_btn = st.sidebar.button("🔍 Predict RUL")

if predict_btn:
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            predicted_rul = result["predicted_RUL"]
            st.metric(label="🔢 Predicted RUL", value=f"{predicted_rul} cycles")
        else:
            st.error("❌ Prediction failed")
    except Exception as e:
        st.error(f"❌ API error: {e}")

# ---------------------------
# 📈 Real-Time Sensor Monitoring Simulation
# ---------------------------
st.markdown("---")
st.subheader("📡 Real-Time Monitoring (Simulated)")

simulate = st.checkbox("Start Simulation", value=False)

if simulate:
    placeholder = st.empty()
    rul_vals = []
    time_steps = []

    for t in range(20):
        # Simulate time progression
        dynamic_payload = {
            "op_setting_1": np.random.uniform(0.5, 1.5),
            "op_setting_2": np.random.uniform(20, 50),
            "op_setting_3": np.random.uniform(500, 600),
        }
        for i in range(1, 22):
            dynamic_payload[f"sensor_measurement_{i}"] = np.random.uniform(0.0, 1.0)

        response = requests.post("http://localhost:8000/predict", json=dynamic_payload)
        if response.status_code == 200:
            pred = response.json()["predicted_RUL"]
            rul_vals.append(pred)
            time_steps.append(t)

        # Plot
        with placeholder.container():
            st.line_chart(pd.DataFrame({"RUL": rul_vals}, index=time_steps))
            st.caption("📉 RUL over simulated time steps")
        time.sleep(1)

# ---------------------------
# 📊 Historical Sensor Trend (Static Example)
# ---------------------------
st.markdown("---")
st.subheader("📊 Sensor Trend Viewer (From Your Dataset)")

try:
    cmapss_df = load_cmapss_dataset("../data/raw/train_FD001.txt")

    selected_unit = st.selectbox("Select Engine Unit", cmapss_df['unit_number'].unique())
    unit_data = cmapss_df[cmapss_df['unit_number'] == selected_unit]

    sensor_cols = [col for col in cmapss_df.columns if 'sensor_measurement' in col]
    sensor_to_plot = st.multiselect("Select sensors to visualize", sensor_cols, default=[sensor_cols[0]])

    for sensor in sensor_to_plot:
        st.line_chart(unit_data.set_index('time_in_cycles')[sensor])
except Exception as e:
    st.warning(f"⚠️ Failed to load or plot historical data: {e}")