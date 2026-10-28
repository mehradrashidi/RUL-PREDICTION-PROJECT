from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def make_tabular_preprocessor():
    # Both imputation and scaling are fit inside the estimator on training rows only.
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
