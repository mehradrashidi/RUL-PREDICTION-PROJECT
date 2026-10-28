def get_feature_columns(df, include_settings=True, drop_constant=True):
    cols = [c for c in df.columns if c.startswith("sensor_") or (include_settings and c.startswith("setting_"))]
    if drop_constant: cols = [c for c in cols if df[c].nunique(dropna=False) > 1]
    if not cols: raise ValueError("No usable sensor or operating-setting columns found.")
    return cols
