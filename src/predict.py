import joblib
import numpy as np

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