from src.data.split_data import split_engine_ids

def test_engine_splits_are_disjoint_and_cover_all_ids():
    train, val, cal = split_engine_ids(range(1, 101), 0.15, 0.15, seed=42)
    assert not (set(train) & set(val))
    assert not (set(train) & set(cal))
    assert not (set(val) & set(cal))
    assert set(train) | set(val) | set(cal) == set(range(1, 101))
