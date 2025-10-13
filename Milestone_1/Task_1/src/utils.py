import pandas as pd

def dataset_info(df: pd.DataFrame):
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Column Names": list(df.columns)
    }
