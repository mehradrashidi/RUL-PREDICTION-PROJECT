from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ["unit_id", "cycle"] + [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]

def load_cmapss_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}. See README.md for setup.")
    df = pd.read_csv(path, sep=r"\s+", header=None).dropna(axis=1, how="all")
    if df.shape[1] != 26:
        raise ValueError(f"Expected 26 columns in {path}; found {df.shape[1]}.")
    df.columns = COLUMNS
    df["unit_id"] = df["unit_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df

def add_training_rul(df, rul_cap=125):
    out = df.copy()
    max_cycle = out.groupby("unit_id")["cycle"].max().rename("_last_cycle")
    out = out.join(max_cycle, on="unit_id")
    out["RUL"] = out["_last_cycle"] - out["cycle"]
    out.drop(columns="_last_cycle", inplace=True)
    if rul_cap is not None:
        out["RUL"] = out["RUL"].clip(upper=int(rul_cap))
    return out

def load_official_test_rul(path, expected_units):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing official test labels: {path}")
    arr = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype=float)
    if len(arr) != expected_units:
        raise ValueError(f"Expected {expected_units} test RUL labels; found {len(arr)}.")
    return arr
