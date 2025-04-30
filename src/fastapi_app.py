from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import *

app = FastAPI(title="Predictive Maintenance API", description="Predict Remaining Useful Life (RUL)", version="1.0")

# Define input schema
class SensorData(BaseModel):
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_measurement_1: float
    sensor_measurement_2: float
    sensor_measurement_3: float
    sensor_measurement_4: float
    sensor_measurement_5: float
    sensor_measurement_6: float
    sensor_measurement_7: float
    sensor_measurement_8: float
    sensor_measurement_9: float
    sensor_measurement_10: float
    sensor_measurement_11: float
    sensor_measurement_12: float
    sensor_measurement_13: float
    sensor_measurement_14: float
    sensor_measurement_15: float
    sensor_measurement_16: float
    sensor_measurement_17: float
    sensor_measurement_18: float
    sensor_measurement_19: float
    sensor_measurement_20: float
    sensor_measurement_21: float

@app.get("/")
def home():
    return {"message": "Welcome to the Predictive Maintenance API!"}

@app.post("/predict")
def predict(data: SensorData):
    sensor_dict = data.dict()
    prediction = predict_rul(sensor_dict)
    return {"predicted_RUL": prediction}

@app.post("/compare")
def compare_models(data: SensorData):
    sensor_dict = data.dict()
    rf_rul = predict_rul_rf(sensor_dict)
    lstm_rul = predict_rul_lstm(sensor_dict)
    return {"RandomForest_RUL": rf_rul, "LSTM_RUL": lstm_rul}