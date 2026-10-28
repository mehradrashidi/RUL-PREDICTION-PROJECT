from sklearn.linear_model import Ridge
from src.data.preprocessing_pipeline import make_tabular_preprocessor

def build_ridge(alpha=10.0):
    return __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline([
        ("preprocess", make_tabular_preprocessor()), ("model", Ridge(alpha=float(alpha)))])
