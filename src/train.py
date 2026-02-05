from __future__ import annotations

import os
from typing import Dict, Any

import pandas as pd

from .config import Config
from .utils import ensure_dir, save_json
from .preprocessing import Preprocessor
from .modeling import group_train_val_split, train_xgb_classifier


def train_pipeline(train_path: str, out_dir: str, cfg: Config) -> Dict[str, Any]:
    ensure_dir(out_dir)

    df = pd.read_csv(train_path)

    pre = Preprocessor(cfg=cfg)
    X, y, groups = pre.fit_transform(df)

    X_tr, X_va, y_tr, y_va = group_train_val_split(
        X, y, groups=groups, test_size=cfg.test_size, random_state=cfg.random_state
    )

    trained, meta = train_xgb_classifier(X_tr, y_tr, X_va, y_va, random_state=cfg.random_state)

    pre_path = os.path.join(out_dir, "preprocessor.joblib")
    model_path = os.path.join(out_dir, "model.joblib")
    meta_path = os.path.join(out_dir, "metadata.json")

    pre.save(pre_path)
    trained.save(model_path)

    le = pre.label_encoder
    label_map = {cls: int(code) for cls, code in zip(le.classes_, le.transform(le.classes_))}
    inv_map = {int(code): cls for cls, code in label_map.items()}

    metadata: Dict[str, Any] = {
        "train_path": train_path,
        "random_state": cfg.random_state,
        "test_size": cfg.test_size,
        "label_map": label_map,
        "inv_label_map": inv_map,
        "threshold": trained.threshold,
        "metrics": meta,
        "n_features": len(pre.feature_names_ or []),
    }
    save_json(metadata, meta_path)

    return metadata
