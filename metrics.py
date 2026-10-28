import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def regression_metrics(y_true,y_pred):
    y=np.asarray(y_true,dtype=float).reshape(-1); p=np.asarray(y_pred,dtype=float).reshape(-1)
    if len(y)!=len(p) or not len(y): raise ValueError("Targets and predictions must be non-empty and same length.")
    return {"MAE":float(mean_absolute_error(y,p)),"RMSE":float(mean_squared_error(y,p)**0.5),"R2":float(r2_score(y,p))}
