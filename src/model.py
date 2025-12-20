from xgboost import XGBClassifier


def build_xgb(params: dict) -> XGBClassifier:
    return XGBClassifier(**params)