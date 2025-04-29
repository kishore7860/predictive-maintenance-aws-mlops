import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

from preprocessing import load_cmapss_dataset, add_rul

def train_model():
    # Load and preprocess data
    data_path = "data/raw/train_FD001.txt"
    df = load_cmapss_dataset(data_path)
    df = add_rul(df)

    # Feature selection (remove IDs and non-sensor info)
    features = df.drop(columns=['unit_number', 'time_in_cycles', 'RUL'])
    labels = df['RUL']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Define and train model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Model Trained - RMSE: {rmse:.2f}, R2: {r2:.2f}")

    # Save model
    model_path = "models/random_forest_model.pkl"
    joblib.dump(model, model_path)
    print(f"📦 Model saved at: {model_path}")

    # MLflow logging
    mlflow.set_experiment("predictive-maintenance")
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(model, "random_forest_model")

if __name__ == "__main__":
    train_model()