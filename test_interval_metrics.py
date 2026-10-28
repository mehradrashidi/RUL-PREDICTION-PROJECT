from src.uncertainty.interval_metrics import interval_metrics
from src.uncertainty.conformal_prediction import conformal_quantile,prediction_interval

def test_interval_metrics_full_coverage():
    r=interval_metrics([1,2,3],[0,1,2],[2,3,4],alpha=.1)
    assert r["Coverage"]==1.0 and r["Mean_Interval_Width"]==2.0

def test_conformal_quantile_and_bounds():
    q=conformal_quantile([1,2,3,4],[1,1,3,2],alpha=.25)
    assert q>=0
    lo,hi=prediction_interval([2,5],q)
    assert all(lo>=0) and all(hi>=lo)
