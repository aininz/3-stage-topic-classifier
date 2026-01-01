import numpy as np
import pandas as pd

def xy_from_df(df: pd.DataFrame, text_col: str, label_col: str):
    X = df[text_col].to_numpy(dtype=object)
    y = df[label_col].to_numpy(dtype=object)
    return X, y