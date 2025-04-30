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
from preprocessing import add_rul


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

# Sensor Trend Viewer with Anomaly Flags
# Section: Sensor Trend Viewer with Anomaly Logging and Summary
st.markdown("---")
st.subheader("📊 Sensor Trend Viewer with RUL Overlay & Anomaly Logging")

try:
    cmapss_df = load_cmapss_dataset("data/raw/train_FD001.txt")
    cmapss_df = add_rul(cmapss_df)

    selected_unit = st.selectbox("Select Engine Unit", cmapss_df['unit_number'].unique())
    unit_data = cmapss_df[cmapss_df['unit_number'] == selected_unit]

    sensor_cols = [col for col in cmapss_df.columns if 'sensor_measurement' in col]
    sensor_to_plot = st.multiselect("Select sensors to visualize", sensor_cols, default=[sensor_cols[0]])

    for sensor in sensor_to_plot:
        st.write(f"### {sensor} vs RUL with Anomaly Logging")

        # 🔧 Slider for threshold percentile
        threshold_pct = st.slider(f"Anomaly threshold for {sensor} (percentile)", 80, 99, 95)
        threshold_value = unit_data[sensor].quantile(threshold_pct / 100)

        # 📊 Prepare Data
        df_plot = unit_data[['time_in_cycles', sensor, 'RUL']].copy()
        anomalies = df_plot[df_plot[sensor] > threshold_value]

        # 🖼️ Plot with anomalies
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df_plot['time_in_cycles'], df_plot[sensor], label=f'{sensor}', color='tab:blue')
        ax1.scatter(anomalies['time_in_cycles'], anomalies[sensor], color='red', label='Anomaly 🔥', marker='o')
        ax1.set_xlabel('Time in Cycles')
        ax1.set_ylabel(sensor, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(df_plot['time_in_cycles'], df_plot['RUL'], label='RUL', color='tab:green')
        ax2.set_ylabel('Remaining Useful Life', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        fig.tight_layout()
        st.pyplot(fig)

        # 📊 Show summary
        if not anomalies.empty:
            st.success(f"✅ {len(anomalies)} anomalies found for {sensor} (>{threshold_value:.2f})")
            st.dataframe(anomalies.describe().T)

            # 💾 Save anomaly log
            output_dir = "logs"
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f"anomalies_unit{selected_unit}_{sensor}.csv")
            anomalies.to_csv(file_path, index=False)
            st.download_button("📥 Download Anomaly Log", data=anomalies.to_csv(index=False), file_name=os.path.basename(file_path), mime="text/csv")

        else:
            st.info("No anomalies found at this threshold.")

except Exception as e:
    st.warning(f"⚠️ Error during anomaly analysis: {e}")

# Dual Model RUL Comparison
st.markdown("---")
st.subheader("🔁 Dual Model Comparison (Random Forest vs LSTM)")

compare_btn = st.button("Compare RUL Predictions")

if compare_btn:
    try:
        response = requests.post("http://localhost:8000/compare", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.metric(label="Random Forest Prediction", value=f"{result['RandomForest_RUL']} cycles")
            st.metric(label="LSTM Prediction", value=f"{result['LSTM_RUL']} cycles")
        else:
            st.error("❌ Failed to compare models")
    except Exception as e:
        st.error(f"❌ API error: {e}")