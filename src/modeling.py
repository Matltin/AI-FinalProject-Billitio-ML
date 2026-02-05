from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupShuffleSplit

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


def group_train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    (tr_idx, va_idx) = next(gss.split(X, y, groups=groups))
    return X.iloc[tr_idx], X.iloc[va_idx], y.iloc[tr_idx], y.iloc[va_idx]


def find_best_threshold(y_true: np.ndarray, proba_pos: np.ndarray) -> Tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (proba_pos >= t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


@dataclass
class TrainedModel:
    model: Any
    threshold: float

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        dump(self, path)

    @staticmethod
    def load(path: str) -> "TrainedModel":
        return load(path)


def train_xgb_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
) -> Tuple[TrainedModel, Dict[str, Any]]:
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed or failed to import. Install it via requirements.txt")

    clf = XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
    )

    # Early stopping اگر نسخه xgboost اجازه دهد
    try:
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=50)
    except TypeError:
        clf.fit(X_train, y_train)

    proba_val = clf.predict_proba(X_val)[:, 1]
    best_t, best_f1 = find_best_threshold(y_val.to_numpy(), proba_val)
    pred_val = (proba_val >= best_t).astype(int)

    report = classification_report(y_val, pred_val, output_dict=True, zero_division=0)
    meta = {
        "val_f1_at_best_threshold": best_f1,
        "best_threshold": best_t,
        "classification_report": report,
    }
    return TrainedModel(model=clf, threshold=best_t), meta
