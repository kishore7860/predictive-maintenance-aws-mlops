import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_prepare_data_lstm(filepath, sequence_length=30):
    # Load and preprocess CMAPSS
    column_names = ['unit_number', 'time_in_cycles'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_measurement_{i}' for i in range(1, 22)]
    df = pd.read_csv(filepath, sep=' ', header=None)
    df.drop(df.columns[[26, 27]], axis=1, inplace=True)
    df.columns = column_names

    # Calculate RUL
    rul = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    rul.columns = ['unit_number', 'max_cycle']
    df = df.merge(rul, on='unit_number', how='left')
    df['RUL'] = df['max_cycle'] - df['time_in_cycles']
    df.drop('max_cycle', axis=1, inplace=True)

    # Normalize sensor values
    features = [f'sensor_measurement_{i}' for i in range(1, 22)]
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    # Reshape into sequences
    sequences = []
    labels = []
    for unit in df['unit_number'].unique():
        unit_df = df[df['unit_number'] == unit]
        unit_array = unit_df[features].values
        rul_array = unit_df['RUL'].values

        for i in range(len(unit_array) - sequence_length):
            sequences.append(unit_array[i:i + sequence_length])
            labels.append(rul_array[i + sequence_length])

    return np.array(sequences), np.array(labels), scaler