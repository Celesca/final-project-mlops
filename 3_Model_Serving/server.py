from fastapi import FastAPI, HTTPException
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import os
import json
import logging
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pydantic import BaseModel

# Ensure repository root is on sys.path so `dags` package can be imported
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from model_serving.model import FraudDetectionModel
from model_serving import preprocessing, data_access, training
from model_serving.schemas import (
    TrainRequest,
    TrainResponse,
    PredictRequest,
    EnrichedTransaction,
    MasterTransaction,
)


app = FastAPI(
    title="Fraud Detection API",
    description="REST API for fraud model training and inference with MLflow tracking",
    version="2.0.0",
)

logger = logging.getLogger("fraud_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Query API routes for predictions
class UpdatePredictionRequest(BaseModel):
    transaction_id: int
    actual_label: bool


def _clean_transaction_data(txn: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive/undeclared fields from transaction_data per request.

    We drop: `isFraud`, `isFlaggedFraud`, `ingest_date`, `source_file`.
    """
    if not txn:
        return {}
    return {k: v for k, v in txn.items() if k not in {"isFraud", "isFlaggedFraud", "ingest_date", "source_file"}}

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

CANONICAL_MODEL_PATH = MODEL_DIR / "xgb_model.joblib"
ARTIFACTS_PATH = MODEL_DIR / "preprocessing_artifacts.joblib"
TRAIN_COLS_PATH = MODEL_DIR / "train_cols.json"
META_PATH = MODEL_DIR / "best_model_meta.json"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "fraud_detection_serving")
MASTER_TABLE_LIMIT = int(os.getenv("MASTER_TABLE_LIMIT", "0"))
BASELINE_SAMPLE_SIZE = int(os.getenv("BASELINE_SAMPLE_SIZE", "5000"))
MODEL_IMPROVEMENT_DELTA = float(os.getenv("MODEL_IMPROVEMENT_DELTA", "0.0"))


@app.on_event("startup")
async def startup_event():
    """Load artifacts and model; establish baseline metric if possible."""
    artifacts = preprocessing.load_preprocessing_artifacts(ARTIFACTS_PATH.as_posix())
    model = FraudDetectionModel()
    if model.model is None and CANONICAL_MODEL_PATH.exists():
        try:
            model.load(CANONICAL_MODEL_PATH.as_posix())
            logger.info("Loaded model from %s", CANONICAL_MODEL_PATH)
        except Exception:
            logger.exception("Failed to load model from %s", CANONICAL_MODEL_PATH)

    meta = _load_model_meta()
    best_metric = meta.get("val_auc") if meta else None

    if best_metric is None and artifacts and model.model is not None:
        try:
            baseline = _evaluate_against_master(model, artifacts, BASELINE_SAMPLE_SIZE)
            if baseline is not None:
                best_metric = baseline
                _save_model_meta({"val_auc": baseline, "run_id": "bootstrap"})
                logger.info("Baseline AUC established at %.4f", baseline)
        except Exception:
            logger.exception("Failed to compute baseline metric")

    app.state.artifacts = artifacts
    app.state.model = model
    app.state.best_metric = best_metric


@app.get("/")
async def root():
    return {
        "message": "Fraud Detection API is running",
        "version": "2.0.0",
        "endpoints": [
            "POST /train",
            "POST /predict",
            "GET /docs",
        ],
    }


@app.post("/train", response_model=TrainResponse)
async def train_endpoint(_: TrainRequest | None = None):
    try:
        master_limit = MASTER_TABLE_LIMIT if MASTER_TABLE_LIMIT > 0 else None
        master_df = data_access.fetch_master_transactions(limit=master_limit)
        if master_df.empty:
            return TrainResponse(
                run_id="N/A",
                val_auc=0.0,
                val_accuracy=0.0,
                promoted=False,
                model_path=None,
                message="No data available for training",
            )

        logger.info("Training on %d master rows from Postgres", len(master_df))

        result = training.train_new_model(
            master_df,
            tracking_uri=MLFLOW_TRACKING_URI,
            experiment_name=MLFLOW_EXPERIMENT,
        )

        promoted = _maybe_promote_model(result)
        message = (
            "New model promoted as serving model"
            if promoted
            else "Model logged to MLflow but not promoted (metric not improved)"
        )

        return TrainResponse(
            run_id=result.run_id,
            val_auc=result.metrics["val_auc"],
            val_accuracy=result.metrics["val_accuracy"],
            promoted=promoted,
            model_path=CANONICAL_MODEL_PATH.as_posix() if promoted else None,
            message=message,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")


@app.post("/predict", response_model=List[EnrichedTransaction])
async def predict_endpoint(request: PredictRequest):
    if not request.new_data:
        raise HTTPException(status_code=400, detail="new_data must not be empty")

    artifacts = getattr(app.state, "artifacts", None)
    model: FraudDetectionModel = getattr(app.state, "model", None)

    if artifacts is None:
        raise HTTPException(status_code=500, detail="Preprocessing artifacts are unavailable")
    if model is None or model.model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        payload_df = _payload_to_dataframe(request.new_data)
        feature_source = preprocessing.build_feature_source_from_master(payload_df)
        features, _ = preprocessing.prepare_feature_frame(feature_source, artifacts=artifacts, fit=False)
        probs = model.predict_probabilities(features)
        preds = (probs >= 0.5).astype(int)

        enriched: List[EnrichedTransaction] = []
        records = payload_df.to_dict(orient="records")
        for record, prob, pred in zip(records, probs, preds):
            enriched.append(
                EnrichedTransaction(
                    **record,
                    predict_proba=float(prob),
                    prediction=int(pred),
                )
            )
        return enriched
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


def _payload_to_dataframe(records: List[MasterTransaction]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    rows = [record.dict() for record in records]
    return pd.DataFrame(rows)


def _maybe_promote_model(result: training.TrainingResult) -> bool:
    candidate_metric = result.metrics.get("val_auc")
    current_best = getattr(app.state, "best_metric", None)
    improved = current_best is None or (candidate_metric > current_best + MODEL_IMPROVEMENT_DELTA)

    if not improved:
        logger.info(
            "New model (AUC=%.4f) did not beat best metric %.4f",
            candidate_metric,
            current_best,
        )
        return False

    joblib.dump(result.model, CANONICAL_MODEL_PATH)
    joblib.dump(result.artifacts, ARTIFACTS_PATH)
    if result.artifacts.get("train_cols"):
        with open(TRAIN_COLS_PATH, "w", encoding="utf-8") as f:
            json.dump(result.artifacts["train_cols"], f)

    _save_model_meta(
        {
            "val_auc": candidate_metric,
            "run_id": result.run_id,
            "updated_at": datetime.utcnow().isoformat(),
        }
    )
    app.state.best_metric = candidate_metric
    app.state.artifacts = result.artifacts
    app.state.model.load(CANONICAL_MODEL_PATH.as_posix())
    logger.info("Promoted new model (AUC=%.4f) from run %s", candidate_metric, result.run_id)
    return True


def _load_model_meta() -> Optional[Dict[str, Any]]:
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_model_meta(meta: Dict[str, Any]) -> None:
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _evaluate_against_master(
    model: FraudDetectionModel,
    artifacts: Dict[str, Any],
    limit: int,
) -> Optional[float]:
    sample_limit = limit if limit > 0 else None
    df = data_access.fetch_master_transactions(limit=sample_limit)
    if df.empty or "isFraud" not in df.columns:
        return None
    labelled = df.dropna(subset=["isFraud"])
    if labelled.empty:
        return None

    feature_source = preprocessing.build_feature_source_from_master(labelled)
    features, _ = preprocessing.prepare_feature_frame(feature_source, artifacts=artifacts, fit=False)
    probs = model.predict_probabilities(features)
    labels = labelled["isFraud"].astype(int).values

    if np.unique(labels).shape[0] < 2:
        return None

    auc = roc_auc_score(labels, probs)
    return float(auc)


@app.get("/query/GET/predictions")
def get_predictions(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return joined predictions from correct and incorrect prediction tables.

    Each record includes prediction metadata and cleaned `transaction_data`.
    """
    from dags.utils import database as db
    
    correct = db.get_correct_predictions(limit)
    incorrect = db.get_incorrect_predictions(limit)

    combined: List[Dict[str, Any]] = []

    for r in correct:
        tx = _clean_transaction_data(r.get("transaction_data") or {})
        combined.append(
            {
                "id": r.get("id"),
                "transaction_id": r.get("transaction_id"),
                "prediction": r.get("prediction"),
                "actual_label": r.get("actual_label"),
                "prediction_time": r.get("prediction_time"),
                "created_at": r.get("created_at"),
                "transaction_data": tx,
                "source": "correct",
            }
        )

    for r in incorrect:
        tx = _clean_transaction_data(r.get("transaction_data") or {})
        combined.append(
            {
                "id": r.get("id"),
                "transaction_id": r.get("transaction_id"),
                "prediction": r.get("prediction"),
                "actual_label": r.get("actual_label"),
                "predict_proba": r.get("predict_proba"),
                "prediction_time": r.get("prediction_time"),
                "created_at": r.get("created_at"),
                "transaction_data": tx,
                "source": "incorrect",
            }
        )

    # sort by prediction_time (descending) when available
    combined.sort(key=lambda x: x.get("prediction_time") or "", reverse=True)
    return combined


@app.get("/query/get/non_frauds")
def get_non_frauds(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return transaction details coming only from `correct_predictions`.

    Transaction details are cleaned to remove sensitive/hidden fields.
    """
    from dags.utils import database as db
    
    correct = db.get_correct_predictions(limit)
    results: List[Dict[str, Any]] = []
    for r in correct:
        tx = _clean_transaction_data(r.get("transaction_data") or {})
        results.append(
            {
                "transaction_id": r.get("transaction_id"),
                "prediction": r.get("prediction"),
                "actual_label": r.get("actual_label"),
                "prediction_time": r.get("prediction_time"),
                "transaction_data": tx,
            }
        )
    return results


@app.get("/query/GET/frauds")
def get_frauds(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return transaction details coming only from `incorrect_predictions` (FP and FN).

    Includes `predict_proba` and cleaned transaction_data.
    """
    from dags.utils import database as db
    
    incorrect = db.get_incorrect_predictions(limit)
    results: List[Dict[str, Any]] = []
    for r in incorrect:
        tx = _clean_transaction_data(r.get("transaction_data") or {})
        results.append(
            {
                "transaction_id": r.get("transaction_id"),
                "prediction": r.get("prediction"),
                "actual_label": r.get("actual_label"),
                "predict_proba": r.get("predict_proba"),
                "prediction_time": r.get("prediction_time"),
                "transaction_data": tx,
            }
        )
    return results


@app.put("/query/PUT/predictions")
def update_prediction_label(payload: UpdatePredictionRequest) -> Dict[str, int]:
    """Update the `actual_label` for a prediction record by `transaction_id`.

    This updates whichever table(s) contain the `transaction_id` (correct_predictions
    and/or incorrect_predictions). It does not move records between tables.
    """
    from dags.utils import database as db
    
    tid = payload.transaction_id
    actual_int = 1 if payload.actual_label else 0

    updated_correct = 0
    updated_incorrect = 0

    # Use the context manager from database.py to run two updates in a single transaction
    with db.get_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE correct_predictions SET actual_label = %s WHERE transaction_id = %s",
            (actual_int, tid),
        )
        updated_correct = cur.rowcount

        cur.execute(
            "UPDATE incorrect_predictions SET actual_label = %s WHERE transaction_id = %s",
            (actual_int, tid),
        )
        updated_incorrect = cur.rowcount

    if (updated_correct + updated_incorrect) == 0:
        raise HTTPException(status_code=404, detail=f"No prediction record found for transaction_id={tid}")

    return {"updated_correct": updated_correct, "updated_incorrect": updated_incorrect}

