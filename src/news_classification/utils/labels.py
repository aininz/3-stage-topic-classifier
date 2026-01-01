    
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target

def infer_label_kind(y: pd.Series) -> bool:
    s = pd.Series(y).dropna()
    if len(s) == 0:
        return False
    if pd.api.types.is_integer_dtype(s):
        return True
    if pd.api.types.is_float_dtype(s) and (s % 1 == 0).all():
        return True
    s_str = s.astype(str).str.strip()
    return bool(s_str.str.fullmatch(r"-?\d+").all())
    
def coerce_labels(y, *, as_str: bool = False) -> np.ndarray:
    y = np.asarray(y).ravel()
    s = pd.Series(y)

    s = s.dropna()

    if as_str:
        s = s.astype(str).str.strip()
        s = s[s.str.len() > 0]
        s = s[s.str.lower() != "nan"]
    else:
        if pd.api.types.is_float_dtype(s):
            if (s % 1 == 0).all():
                s = s.astype("int64")
            else:
                s = s.astype(str).str.strip()

    y_out = s.to_numpy()

    tot = type_of_target(y_out)
    if tot not in {"binary", "multiclass"}:
        raise ValueError(f"Bad label target type: {tot}")

    return y_out