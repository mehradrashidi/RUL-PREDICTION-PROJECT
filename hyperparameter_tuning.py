import itertools
import pandas as pd
from sklearn.metrics import mean_absolute_error
from src.models.ridge_model import build_ridge
from src.models.lightgbm_model import build_lightgbm

def grid_search_tabular(X_train,y_train,X_val,y_val,model_name,param_grid):
    """Select parameters on validation data only; official test is never passed here."""
    keys=list(param_grid); rows=[]; best=None; best_score=float("inf"); best_params=None
    for vals in itertools.product(*(param_grid[k] for k in keys)):
        params=dict(zip(keys,vals))
        if model_name.lower()=="ridge": model=build_ridge(alpha=params["alpha"])
        elif model_name.lower()=="lightgbm": model=build_lightgbm(**params)
        else: raise ValueError("Supported models: ridge, lightgbm")
        model.fit(X_train,y_train); pred=model.predict(X_val); score=float(mean_absolute_error(y_val,pred))
        rows.append({"model":model_name,**params,"validation_MAE":score})
        if score<best_score: best,best_score,best_params=model,score,params
    return best,best_params,pd.DataFrame(rows).sort_values("validation_MAE").reset_index(drop=True)
