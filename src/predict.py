import joblib
import numpy as np
import tensorflow as tf

# Load the trained model
model = joblib.load("models/random_forest_model.pkl")

def predict_rul(sensor_data: dict) -> float:
    """
    Predict Remaining Useful Life (RUL) based on sensor readings.
    
    Args:
        sensor_data (dict): Dictionary containing sensor values.
    
    Returns:
        float: Predicted RUL
    """
    # Convert dictionary to the correct feature array
    feature_order = [f'op_setting_{i}' for i in range(1,4)] + [f'sensor_measurement_{i}' for i in range(1,22)]
    input_features = np.array([sensor_data[feature] for feature in feature_order]).reshape(1, -1)

    # Make prediction
    predicted_rul = model.predict(input_features)[0]
    return round(predicted_rul, 2)


lstm_model = tf.keras.models.load_model("models/lstm_model.keras")

# Inference functions
def predict_rul_rf(sensor_data: dict) -> float:
    feature_order = [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_measurement_{i}' for i in range(1, 22)]
    input_features = np.array([sensor_data[feature] for feature in feature_order]).reshape(1, -1)
    return float(round(model.predict(input_features)[0], 2))

def predict_rul_lstm(sensor_data: dict) -> float:
    sensor_only = [sensor_data[f'sensor_measurement_{i}'] for i in range(1, 22)]
    sensor_scaled = np.array(sensor_only).reshape((1, 1, 21))  # [samples, timesteps, features]
    prediction = lstm_model.predict(sensor_scaled)[0][0]
    return float(round(prediction, 2))

