from lightgbm import LGBMRegressor

def build_lightgbm(**params):
    defaults=dict(n_estimators=400, learning_rate=0.04, num_leaves=31, max_depth=-1,
                  subsample=0.9, colsample_bytree=0.9, random_state=42, verbosity=-1, n_jobs=-1)
    defaults.update(params)
    return LGBMRegressor(**defaults)
