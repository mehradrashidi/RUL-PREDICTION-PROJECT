import numpy as np

def conformal_quantile(y_cal,pred_cal,alpha=0.10):
    """Finite-sample split-conformal quantile of absolute calibration residuals."""
    y=np.asarray(y_cal,dtype=float).reshape(-1); p=np.asarray(pred_cal,dtype=float).reshape(-1)
    if len(y)!=len(p) or not len(y): raise ValueError("Calibration arrays must be non-empty and equal length.")
    if not 0<alpha<1: raise ValueError("alpha must be between 0 and 1.")
    scores=np.sort(np.abs(y-p)); rank=int(np.ceil((len(scores)+1)*(1-alpha)))
    if rank>len(scores): return float("inf")
    return float(scores[rank-1])

def prediction_interval(pred,q,lower_bound=0.0):
    p=np.asarray(pred,dtype=float).reshape(-1)
    if np.isinf(q): return np.full_like(p,lower_bound),np.full_like(p,np.inf)
    lo=np.maximum(lower_bound,p-q); hi=np.maximum(lo,p+q)
    return lo,hi
