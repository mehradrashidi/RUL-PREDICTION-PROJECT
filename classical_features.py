import pandas as pd

def make_last_cycle_features(df, feature_cols):
    latest = df.sort_values(["unit_id", "cycle"]).groupby("unit_id", as_index=False).tail(1)
    return latest[["unit_id"] + list(feature_cols)].sort_values("unit_id").reset_index(drop=True)

def make_window_summary_features(df, feature_cols, window_size=30):
    rows=[]
    for uid, group in df.sort_values("cycle").groupby("unit_id"):
        g=group.tail(window_size); row={"unit_id": int(uid)}
        for col in feature_cols:
            v=g[col].astype(float).to_numpy()
            row[f"{col}__last"]=float(v[-1]); row[f"{col}__mean"]=float(v.mean())
            row[f"{col}__std"]=float(v.std(ddof=0)); row[f"{col}__slope"]=float((v[-1]-v[0])/max(len(v)-1,1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)
