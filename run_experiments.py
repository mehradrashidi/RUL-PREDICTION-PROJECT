from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.utils.reproducibility import set_seed
from src.data.prepare_dataset import prepare_data
from src.data.preprocess import data_quality_report
from src.data.sequence_builder import make_sequences
from src.training.hyperparameter_tuning import grid_search_tabular
from src.training.train_lstm import train_lstm
from src.evaluation.metrics import regression_metrics
from src.evaluation.plots import plot_model_comparison,plot_predictions,plot_residuals
from src.uncertainty.conformal_prediction import conformal_quantile,prediction_interval
from src.uncertainty.interval_metrics import interval_metrics

def run(config_path="configs/config.yaml"):
    with open(config_path,encoding="utf-8") as f: cfg=yaml.safe_load(f)
    set_seed(cfg["project"].get("random_seed",42)); out=Path("outputs")
    for d in ["figures","tables","predictions"]:(out/d).mkdir(parents=True,exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    d=prepare_data(cfg); feats=d["feature_cols"]; tr=d["train_part"]; va=d["val_part"]; ca=d["cal_part"]
    data_quality_report(d["train_raw"]).to_csv(out/"tables/data_quality.csv",index=False)
    # Tabular model selection uses all labelled cycles from train engines; validation engines are disjoint.
    Xtr,ytr=tr[feats],tr["RUL"]; Xva,yva=va[feats],va["RUL"]
    ridge,ridge_params,ridge_grid=grid_search_tabular(Xtr,ytr,Xva,yva,"ridge",{"alpha":cfg["models"].get("ridge_alphas",[1,10,50])})
    lgb_grid_cfg=cfg["models"].get("lightgbm_grid",{})
    lgb_grid_cfg={k:list(v) for k,v in lgb_grid_cfg.items()}
    lgb_grid_cfg.update({"random_state":[cfg["project"].get("random_seed",42)],"verbosity":[-1],"n_jobs":[-1]})
    lgbm,lgbm_params,lgbm_grid=grid_search_tabular(Xtr,ytr,Xva,yva,"lightgbm",lgb_grid_cfg)
    ridge_grid.to_csv(out/"tables/ridge_tuning.csv",index=False); lgbm_grid.to_csv(out/"tables/lightgbm_tuning.csv",index=False)
    joblib.dump(ridge,"models/ridge_best.joblib"); joblib.dump(lgbm,"models/lightgbm_best.joblib")
    # LSTM scaler/imputer are fit only on training-engine rows, then applied to all other splits.
    imputer=SimpleImputer(strategy="median"); scaler=StandardScaler(); imputer.fit(tr[feats]); scaler.fit(imputer.transform(tr[feats]))
    def scaled(frame):
        z=frame.copy(); z.loc[:,feats]=scaler.transform(imputer.transform(frame[feats])); return z
    tr_s,va_s,ca_s,te_s=map(scaled,[tr,va,ca,d["test_raw"]]); win=int(cfg["data"].get("window_size",30))
    Xtr_seq,ytr_seq,_=make_sequences(tr_s,feats,win,"RUL",False)
    Xva_seq,yva_seq,_=make_sequences(va_s,feats,win,"RUL",False)
    model,history=train_lstm(Xtr_seq,ytr_seq,Xva_seq,yva_seq,cfg["models"]["lstm"])
    pd.DataFrame(history.history).to_csv(out/"tables/lstm_training_history.csv",index=False)
    # Conformal calibration and final evaluation use one last-cycle prediction per engine for every model.
    ca_latest=ca.sort_values(["unit_id","cycle"]).groupby("unit_id",as_index=False).tail(1).sort_values("unit_id")
    Xca_lstm, yca_lstm, ca_meta=make_sequences(ca_s,feats,win,"RUL",True)
    test_latest=d["test_latest"]; Xte_lstm,te_meta=make_sequences(te_s,feats,win,None,True)
    if not np.array_equal(ca_latest.unit_id.to_numpy(),ca_meta.unit_id.to_numpy()): raise RuntimeError("Calibration engine order mismatch.")
    if not np.array_equal(test_latest.unit_id.to_numpy(),te_meta.unit_id.to_numpy()): raise RuntimeError("Test engine order mismatch.")
    y_test=d["test_rul"]
    preds={"Ridge":ridge.predict(test_latest[feats]),"LightGBM":lgbm.predict(test_latest[feats]),"LSTM":model.predict(Xte_lstm,verbose=0).reshape(-1)}
    cal_preds={"Ridge":ridge.predict(ca_latest[feats]),"LightGBM":lgbm.predict(ca_latest[feats]),"LSTM":model.predict(Xca_lstm,verbose=0).reshape(-1)}
    cal_y={"Ridge":ca_latest["RUL"].to_numpy(),"LightGBM":ca_latest["RUL"].to_numpy(),"LSTM":yca_lstm}
    alpha=float(cfg["uncertainty"].get("alpha",.10)); result_rows=[]
    pred_table=pd.DataFrame({"unit_id":test_latest.unit_id.to_numpy(),"actual_RUL":y_test})
    for name,pred in preds.items():
        q=conformal_quantile(cal_y[name],cal_preds[name],alpha); lo,hi=prediction_interval(pred,q)
        row={"model":name,**regression_metrics(y_test,pred),**interval_metrics(y_test,lo,hi,alpha),"conformal_q":q}
        result_rows.append(row); pred_table[f"{name}_prediction"]=pred; pred_table[f"{name}_lower"]=lo; pred_table[f"{name}_upper"]=hi
        plot_residuals(y_test,pred,out/f"figures/{name.lower()}_test_residuals.png",f"{name}: test residuals")
    results=pd.DataFrame(result_rows).sort_values("MAE").reset_index(drop=True)
    results.to_csv(out/"tables/final_model_comparison.csv",index=False); pred_table.to_csv(out/"predictions/test_predictions_with_intervals.csv",index=False)
    plot_model_comparison(results,out/"figures/model_comparison.png"); plot_predictions(y_test,preds,out/"figures/actual_vs_predicted.png")
    with open(out/"tables/best_params.json","w",encoding="utf-8") as f: json.dump({"Ridge":ridge_params,"LightGBM":lgbm_params},f,indent=2)
    print("\nFinal test results (actual metrics from this run):\n",results.to_string(index=False))
    return results
