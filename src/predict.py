from __future__ import annotations

import os
import pandas as pd

from .utils import load_json, ensure_dir
from .preprocessing import Preprocessor
from .modeling import TrainedModel


def predict_pipeline(
    test_path: str,
    artifacts_dir: str,
    output_path: str,
    ticket_id_col: str = "TicketID",
) -> str:
    ensure_dir(os.path.dirname(output_path) or ".")

    pre_path = os.path.join(artifacts_dir, "preprocessor.joblib")
    model_path = os.path.join(artifacts_dir, "model.joblib")
    meta_path = os.path.join(artifacts_dir, "metadata.json")

    pre = Preprocessor.load(pre_path)
    trained = TrainedModel.load(model_path)
    meta = load_json(meta_path)

    df_test = pd.read_csv(test_path)

    if ticket_id_col not in df_test.columns:
        raise ValueError(f"Test file must contain '{ticket_id_col}' to create submission.")

    X_test = pre.transform(df_test)
    pred_num = trained.predict(X_test)

    inv_map = {int(k): v for k, v in meta["inv_label_map"].items()}
    pred_label = [inv_map[int(p)] for p in pred_num]

    submission = pd.DataFrame({
        ticket_id_col: df_test[ticket_id_col].values,
        "TripReason": pred_label
    })

    submission.to_csv(output_path, index=False)
    return output_path
