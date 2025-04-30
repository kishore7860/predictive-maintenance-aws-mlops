import pandas as pd
import numpy

def load_cmapss_dataset(filepath):
    """Load and preprocess CMAPSS dataset."""
    column_names = ['unit_number', 'time_in_cycles'] + [f'op_setting_{i}' for i in range(1,4)] + [f'sensor_measurement_{i}' for i in range(1,22)]
    df = pd.read_csv(filepath, sep=' ', header=None)
    df.drop(df.columns[[26,27]], axis=1, inplace=True)
    df.columns = column_names
    return df

def add_rul(df):
    """Add Remaining Useful Life (RUL) label."""
    rul = pd.DataFrame(df.groupby('unit_number')['time_in_cycles'].max()).reset_index()
    rul.columns = ['unit_number', 'max_cycle']
    df = df.merge(rul, on=['unit_number'], how='left')
    df['RUL'] = df['max_cycle'] - df['time_in_cycles']
    df.drop('max_cycle', axis=1, inplace=True)
    return df

if __name__ == "__main__":
    data_path = "data/raw/train_FD001.txt"
    df = load_cmapss_dataset(data_path)
    df = add_rul(df)
    print(df.head())