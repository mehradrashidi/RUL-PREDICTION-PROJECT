import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def split_engine_ids(unit_ids, validation_fraction=0.15, calibration_fraction=0.15, seed=42):
    ids = np.asarray(sorted(set(unit_ids)))
    if len(ids) < 5: raise ValueError("At least five distinct engines are recommended for grouped splitting.")
    if validation_fraction <= 0 or calibration_fraction < 0 or validation_fraction + calibration_fraction >= 1:
        raise ValueError("Require validation_fraction > 0, calibration_fraction >= 0, and their sum < 1.")
    hold_fraction = validation_fraction + calibration_fraction
    first = GroupShuffleSplit(n_splits=1, test_size=hold_fraction, random_state=seed)
    tr_idx, hold_idx = next(first.split(np.zeros((len(ids), 1)), groups=ids))
    train_ids, hold_ids = ids[tr_idx], ids[hold_idx]
    if calibration_fraction == 0: return train_ids, hold_ids, np.array([], dtype=int)
    cal_share = calibration_fraction / hold_fraction
    second = GroupShuffleSplit(n_splits=1, test_size=cal_share, random_state=seed+1)
    val_idx, cal_idx = next(second.split(np.zeros((len(hold_ids), 1)), groups=hold_ids))
    return train_ids, hold_ids[val_idx], hold_ids[cal_idx]
