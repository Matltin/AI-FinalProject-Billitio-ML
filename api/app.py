import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocessing import Preprocessor
from src.modeling import TrainedModel
from src.utils import load_json

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")

pre = Preprocessor.load(os.path.join(ARTIFACTS_DIR, "preprocessor.joblib"))
trained = TrainedModel.load(os.path.join(ARTIFACTS_DIR, "model.joblib"))
meta = load_json(os.path.join(ARTIFACTS_DIR, "metadata.json"))

inv_map = {int(k): v for k, v in meta["inv_label_map"].items()}

app = FastAPI(title="TripReason API", version="1.0")


class PredictRequest(BaseModel):
    records: list[dict] = Field(..., min_length=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        df = pd.DataFrame(req.records)
        X = pre.transform(df)

        proba = trained.predict_proba(X)  # proba for class 1
        pred_num = (proba >= trained.threshold).astype(int)
        pred_label = [inv_map[int(p)] for p in pred_num]

        return {
            "predictions": pred_label,
            "probabilities": [float(x) for x in proba],
            "threshold": round(trained.threshold, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
