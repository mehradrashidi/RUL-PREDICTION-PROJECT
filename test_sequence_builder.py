import pandas as pd
from src.data.sequence_builder import make_sequences

def test_sequences_do_not_cross_engine_boundaries():
    df=pd.DataFrame({"unit_id":[1,1,1,2,2,2],"cycle":[1,2,3,1,2,3],"x":[10,20,30,100,200,300],"RUL":[2,1,0,2,1,0]})
    X,y,meta=make_sequences(df,["x"],2,"RUL",False)
    assert X.shape==(4,2,1); assert list(y)==[1,0,1,0]; assert list(meta.unit_id)==[1,1,2,2]
    assert X[1,-1,0]==30 and X[3,-1,0]==300

def test_short_engine_is_left_padded():
    df=pd.DataFrame({"unit_id":[1],"cycle":[1],"x":[5.0],"RUL":[3]})
    X,y,meta=make_sequences(df,["x"],3,"RUL",True)
    assert X.shape==(1,3,1); assert X[0,:,0].tolist()==[5,5,5]; assert y.tolist()==[3]
