import numpy as np

def interval_metrics(y_true,lower,upper,alpha=0.10):
    y=np.asarray(y_true,dtype=float).reshape(-1); lo=np.asarray(lower,dtype=float).reshape(-1); hi=np.asarray(upper,dtype=float).reshape(-1)
    if not (len(y)==len(lo)==len(hi)) or not len(y): raise ValueError("Inputs must be non-empty and equal length.")
    if np.any(hi<lo): raise ValueError("Upper bounds must not be below lower bounds.")
    if not 0<alpha<1: raise ValueError("alpha must be between 0 and 1.")
    covered=(y>=lo)&(y<=hi); width=hi-lo
    penalty=(2/alpha)*(lo-y)*(y<lo)+(2/alpha)*(y-hi)*(y>hi)
    return {"Coverage":float(covered.mean()),"Mean_Interval_Width":float(width.mean()),
            "Median_Interval_Width":float(np.median(width)),"Interval_Score":float(np.mean(width+penalty))}
