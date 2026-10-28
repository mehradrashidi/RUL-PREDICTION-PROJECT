import numpy as np
import pandas as pd

def data_quality_report(df):
    rows = []
    for col in df.columns:
        s = df[col]
        inf_count = int(np.isinf(s.to_numpy(dtype=float)).sum()) if pd.api.types.is_numeric_dtype(s) else 0
        rows.append({"column": col, "dtype": str(s.dtype), "missing_count": int(s.isna().sum()),
                     "missing_pct": float(100*s.isna().mean()), "unique_count": int(s.nunique(dropna=False)),
                     "infinite_count": inf_count})
    return pd.DataFrame(rows)
