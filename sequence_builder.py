import numpy as np
import pandas as pd

def make_sequences(df, feature_cols, window_size, target_col="RUL", final_only=False):
    """Build windows within each engine. A window target belongs to its final cycle."""
    if window_size < 1: raise ValueError("window_size must be >= 1")
    xs=[]; ys=[]; units=[]; cycles=[]
    for uid, group in df.sort_values(["unit_id", "cycle"]).groupby("unit_id", sort=True):
        arr=group[feature_cols].to_numpy(dtype=np.float32); cyc=group["cycle"].to_numpy()
        target=group[target_col].to_numpy(dtype=np.float32) if target_col and target_col in group else None
        if len(arr)<window_size:
            pad=window_size-len(arr); arr=np.concatenate([np.repeat(arr[:1],pad,axis=0),arr],axis=0)
            cyc=np.concatenate([np.repeat(cyc[:1],pad),cyc])
            if target is not None: target=np.concatenate([np.repeat(target[:1],pad),target])
        starts=[len(arr)-window_size] if final_only else range(len(arr)-window_size+1)
        for start in starts:
            end=start+window_size-1; xs.append(arr[start:end+1]); units.append(int(uid)); cycles.append(int(cyc[end]))
            if target is not None: ys.append(float(target[end]))
    X=np.asarray(xs,dtype=np.float32); meta=pd.DataFrame({"unit_id":units,"end_cycle":cycles})
    if target_col and len(ys): return X,np.asarray(ys,dtype=np.float32),meta
    return X,meta
