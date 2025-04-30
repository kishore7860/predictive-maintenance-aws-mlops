import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from lstm_utils import load_and_prepare_data_lstm

# Load CMAPSS data
sequence_length = 30
X, y, scaler = load_and_prepare_data_lstm("data/raw/train_FD001.txt", sequence_length=sequence_length)

# Split data
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Define model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(
    loss=MeanSquaredError(),
    optimizer='adam',
    metrics=[MeanAbsoluteError()]
)

# Setup MLflow
mlflow.set_experiment("predictive-maintenance-lstm")
mlflow.tensorflow.autolog()  # enables automatic tracking

with mlflow.start_run() as run:
    print(f"🚀 Tracking run at: {run.info.run_id}")

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=64
    )

    # Manually log parameters (optional)
    mlflow.log_param("sequence_length", sequence_length)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("epochs", 10)

    # Save model locally
    model.save("models/lstm_model.keras")
    mlflow.keras.log_model(model, "lstm_model")

    print("✅ LSTM training complete and logged to MLflow.")