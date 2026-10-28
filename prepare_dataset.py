from src.data.load_data import load_cmapss_file,add_training_rul,load_official_test_rul
from src.data.split_data import split_engine_ids
from src.features.feature_engineering import get_feature_columns

def prepare_data(config):
    train_raw=load_cmapss_file(config["data"]["train_path"]); test_raw=load_cmapss_file(config["data"]["test_path"])
    train=add_training_rul(train_raw,config["data"].get("rul_cap",125))
    features=get_feature_columns(train,config["features"].get("include_settings",True),config["features"].get("drop_constant_sensors",True))
    tr_ids,va_ids,ca_ids=split_engine_ids(train.unit_id.unique(),config["data"].get("validation_fraction",.15),
        config["data"].get("calibration_fraction",.15),config["project"].get("random_seed",42))
    train_part=train[train.unit_id.isin(tr_ids)].copy(); val_part=train[train.unit_id.isin(va_ids)].copy(); cal_part=train[train.unit_id.isin(ca_ids)].copy()
    test_latest=test_raw.sort_values(["unit_id","cycle"]).groupby("unit_id",as_index=False).tail(1).sort_values("unit_id").reset_index(drop=True)
    test_rul=load_official_test_rul(config["data"]["rul_path"],test_latest.unit_id.nunique())
    return {"train_raw":train_raw,"train":train,"train_part":train_part,"val_part":val_part,"cal_part":cal_part,
            "test_raw":test_raw,"test_latest":test_latest,"test_rul":test_rul,"feature_cols":features}
