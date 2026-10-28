from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_model_comparison(results,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    metrics=[m for m in ["MAE","RMSE"] if m in results.columns]
    if not metrics:return
    fig,axes=plt.subplots(1,len(metrics),figsize=(6*len(metrics),4)); axes=np.atleast_1d(axes)
    for ax,m in zip(axes,metrics):
        ax.bar(results["model"],results[m]); ax.set_title(f"{m} by model"); ax.set_ylabel(m); ax.tick_params(axis="x",rotation=20); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def plot_predictions(y_true,pred_map,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); y=np.asarray(y_true,dtype=float)
    fig,ax=plt.subplots(figsize=(7,6))
    for name,pred in pred_map.items(): ax.scatter(y,pred,s=18,alpha=.65,label=name)
    vals=np.concatenate([y]+[np.asarray(p).reshape(-1) for p in pred_map.values()]); lim=[float(np.nanmin(vals)),float(np.nanmax(vals))]
    ax.plot(lim,lim,linestyle="--",label="Ideal"); ax.set_xlabel("Actual RUL"); ax.set_ylabel("Predicted RUL"); ax.set_title("Actual vs predicted RUL"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def plot_residuals(y_true,pred,path,title="Residual analysis"):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); y=np.asarray(y_true); p=np.asarray(pred); r=y-p
    fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(p,r,alpha=.65); ax.axhline(0,linestyle="--"); ax.set_xlabel("Predicted RUL"); ax.set_ylabel("Actual - predicted"); ax.set_title(title); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)
