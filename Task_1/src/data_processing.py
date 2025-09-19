import pandas as pd

def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.json'):
        return pd.read_json(file)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or JSON.")
