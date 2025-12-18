from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Config:
    # paths
    train_path: str = "data/train_data.csv"
    test_path: str = "data/test_data.csv"
    model_path: str = "models/xgb_pipeline.joblib"
    submission_path: str = "outputs/submission.csv"

    # target
    target_col: str = "TripReason"

    # split
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True

    # metric
    f1_average: str = "macro"  # "binary" or "macro"
    positive_label: str = "Work"  # if you use binary f1

    # XGBoost params (baseline)
    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 800,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
    })

    # optional: early stopping
    early_stopping_rounds: Optional[int] = 50
