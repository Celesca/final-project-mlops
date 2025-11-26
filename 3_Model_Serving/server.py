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
from model_serving import preprocessing, data_access, training, training_metadata
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
TRAINING_DATA_LIMIT = int(os.getenv("TRAINING_DATA_LIMIT", "20000"))


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
    """
    Train a new fraud detection model.
    
    Training data source:
    1. Fetches from predictions table (fraud cases first, then non-fraud)
    2. Uses actual_label if available (human-labeled), otherwise uses prediction
    3. Limited to TRAINING_DATA_LIMIT rows (default 20,000)
    
    Falls back to master transactions table if no predictions exist.
    """
    try:
        # First, try to get training data from predictions table
        # This prioritizes fraud cases and uses actual_label when available
        training_df = data_access.fetch_training_data_from_predictions(
            total_limit=TRAINING_DATA_LIMIT
        )
        
        if training_df.empty:
            # Fallback to master transactions if no predictions exist
            logger.info("No predictions found, falling back to master transactions")
            master_limit = MASTER_TABLE_LIMIT if MASTER_TABLE_LIMIT > 0 else TRAINING_DATA_LIMIT
            training_df = data_access.fetch_master_transactions(limit=master_limit)
        
        if training_df.empty:
            return TrainResponse(
                run_id="N/A",
                val_auc=0.0,
                val_accuracy=0.0,
                promoted=False,
                model_path=None,
                message="No data available for training",
            )

        # Log training data composition
        fraud_count = training_df['isFraud'].sum() if 'isFraud' in training_df.columns else 0
        non_fraud_count = len(training_df) - fraud_count
        logger.info(
            "Training on %d rows from predictions table (fraud: %d, non-fraud: %d)",
            len(training_df), fraud_count, non_fraud_count
        )

        result = training.train_new_model(
            training_df,
            tracking_uri=MLFLOW_TRACKING_URI,
            experiment_name=MLFLOW_EXPERIMENT,
        )

        # Compare new model vs current model on the same training data
        promotion_result = _maybe_promote_model(result, training_df=training_df)
        promoted = promotion_result["promoted"]
        message = promotion_result["message"]
        
        # Save training date (latest ingest_date from training data)
        if 'ingest_date' in training_df.columns and not training_df['ingest_date'].isna().all():
            # Get the latest (max) ingest_date from training data
            latest_training_date = training_df['ingest_date'].dropna().max()
            if latest_training_date:
                # Convert to string if it's not already
                if hasattr(latest_training_date, 'strftime'):
                    latest_training_date = latest_training_date.strftime('%Y-%m-%d')
                elif isinstance(latest_training_date, str):
                    # If it's already a string, use it directly
                    pass
                else:
                    latest_training_date = str(latest_training_date)
                
                training_metadata.save_training_date(latest_training_date)
                logger.info(f"Saved training date: {latest_training_date}")
        else:
            logger.warning("No ingest_date found in training data, cannot save training date")
        
        # Add training data info to message
        message += f" Trained on {len(training_df)} rows ({fraud_count} fraud, {non_fraud_count} non-fraud)."

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


def _maybe_promote_model(
    result: training.TrainingResult,
    training_df: pd.DataFrame = None,
) -> Dict[str, Any]:
    """
    Compare new model against current model on the SAME training data.
    Always logs to MLflow. Only promotes if new model is better.
    
    Returns dict with:
        - promoted: bool
        - new_auc: float
        - current_auc: float (or None if no current model)
        - message: str
    """
    candidate_metric = result.metrics.get("val_auc")
    current_model: FraudDetectionModel = getattr(app.state, "model", None)
    current_artifacts = getattr(app.state, "artifacts", None)
    
    # Evaluate current model on the same training data for fair comparison
    current_auc = None
    if current_model is not None and current_model.model is not None and current_artifacts and training_df is not None:
        try:
            labelled_df = training_df.dropna(subset=["isFraud"]).copy()
            if not labelled_df.empty:
                labelled_df["isFraud"] = labelled_df["isFraud"].astype(int)
                feature_source = preprocessing.build_feature_source_from_master(labelled_df)
                features, _ = preprocessing.prepare_feature_frame(
                    feature_source, artifacts=current_artifacts, fit=False
                )
                probs = current_model.predict_probabilities(features)
                labels = labelled_df["isFraud"].values
                if np.unique(labels).shape[0] >= 2:
                    current_auc = roc_auc_score(labels, probs)
                    logger.info("Current model AUC on training data: %.4f", current_auc)
        except Exception as e:
            logger.warning("Failed to evaluate current model on training data: %s", e)
            current_auc = None
    
    # Determine if improvement occurred
    if current_auc is None:
        # No current model or evaluation failed - new model wins
        improved = True
        logger.info("No valid current model evaluation, promoting new model (AUC=%.4f)", candidate_metric)
    else:
        improved = candidate_metric > current_auc + MODEL_IMPROVEMENT_DELTA
        logger.info(
            "Comparison: New model AUC=%.4f vs Current model AUC=%.4f (delta=%.4f)",
            candidate_metric, current_auc, MODEL_IMPROVEMENT_DELTA
        )

    if not improved:
        logger.info(
            "New model (AUC=%.4f) did not beat current model (AUC=%.4f)",
            candidate_metric,
            current_auc,
        )
        return {
            "promoted": False,
            "new_auc": candidate_metric,
            "current_auc": current_auc,
            "message": f"New model AUC ({candidate_metric:.4f}) did not beat current model AUC ({current_auc:.4f}). Model saved to MLflow (run_id: {result.run_id}) but not promoted.",
        }

    # Promote the new model
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
    
    return {
        "promoted": True,
        "new_auc": candidate_metric,
        "current_auc": current_auc,
        "message": f"New model promoted (AUC: {candidate_metric:.4f} vs previous: {current_auc if current_auc else 'N/A'}). Run ID: {result.run_id}",
    }


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
    """Return all predictions from the predictions table.

    Each record includes prediction metadata and transaction data.
    actual_label may be NULL for unlabeled predictions.
    """
    from dags.utils import database as db
    
    predictions = db.get_all_predictions(limit=limit)
    
    results: List[Dict[str, Any]] = []
    for r in predictions:
        results.append({
            "id": r.get("prediction_id"),
            "transaction_id": r.get("transaction_id"),
            "prediction": r.get("prediction"),
            "actual_label": r.get("actual_label"),
            "predict_proba": r.get("predict_proba"),
            "prediction_time": r.get("prediction_time"),
            "created_at": r.get("prediction_created_at"),
            "step": r.get("step"),
            "type": r.get("type"),
            "amount": r.get("amount"),
            "nameOrig": r.get("nameOrig"),
            "oldbalanceOrg": r.get("oldbalanceOrg"),
            "newbalanceOrig": r.get("newbalanceOrig"),
            "nameDest": r.get("nameDest"),
            "oldbalanceDest": r.get("oldbalanceDest"),
            "newbalanceDest": r.get("newbalanceDest"),
            "isFraud": r.get("isFraud"),
            "isFlaggedFraud": r.get("isFlaggedFraud"),
        })
    
    return results


@app.get("/query/GET/non_frauds")
def get_non_frauds_route(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return predictions where prediction = False (predicted as non-fraud).

    actual_label may be NULL for unlabeled predictions.
    """
    from dags.utils import database as db
    
    predictions = db.get_non_frauds(limit=limit)
    
    results: List[Dict[str, Any]] = []
    for r in predictions:
        results.append({
            "id": r.get("prediction_id"),
            "transaction_id": r.get("transaction_id"),
            "prediction": r.get("prediction"),
            "actual_label": r.get("actual_label"),
            "predict_proba": r.get("predict_proba"),
            "prediction_time": r.get("prediction_time"),
            "step": r.get("step"),
            "type": r.get("type"),
            "amount": r.get("amount"),
            "nameOrig": r.get("nameOrig"),
            "oldbalanceOrg": r.get("oldbalanceOrg"),
            "newbalanceOrig": r.get("newbalanceOrig"),
            "nameDest": r.get("nameDest"),
            "oldbalanceDest": r.get("oldbalanceDest"),
            "newbalanceDest": r.get("newbalanceDest"),
            "isFraud": r.get("isFraud"),
            "isFlaggedFraud": r.get("isFlaggedFraud"),
        })
    return results


@app.get("/query/GET/frauds")
def get_frauds_route(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return predictions where prediction = True (predicted as fraud).

    actual_label may be NULL for unlabeled predictions.
    """
    from dags.utils import database as db
    
    predictions = db.get_frauds(limit=limit)
    
    results: List[Dict[str, Any]] = []
    for r in predictions:
        results.append({
            "id": r.get("prediction_id"),
            "transaction_id": r.get("transaction_id"),
            "prediction": r.get("prediction"),
            "actual_label": r.get("actual_label"),
            "predict_proba": r.get("predict_proba"),
            "prediction_time": r.get("prediction_time"),
            "step": r.get("step"),
            "type": r.get("type"),
            "amount": r.get("amount"),
            "nameOrig": r.get("nameOrig"),
            "oldbalanceOrg": r.get("oldbalanceOrg"),
            "newbalanceOrig": r.get("newbalanceOrig"),
            "nameDest": r.get("nameDest"),
            "oldbalanceDest": r.get("oldbalanceDest"),
            "newbalanceDest": r.get("newbalanceDest"),
            "isFraud": r.get("isFraud"),
            "isFlaggedFraud": r.get("isFlaggedFraud"),
        })
    return results


@app.put("/query/PUT/predictions")
def update_prediction_label(payload: UpdatePredictionRequest) -> Dict[str, Any]:
    """Update the `actual_label` for a prediction record by `prediction_id`.

    This is used after human review to label a prediction.
    """
    from dags.utils import database as db
    
    prediction_id = payload.transaction_id  # Using transaction_id field as prediction_id
    actual_label = payload.actual_label

    success = db.update_actual_label(prediction_id, actual_label)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"No prediction record found with id={prediction_id}")

    return {"updated": True, "prediction_id": prediction_id, "actual_label": actual_label}


@app.get("/query/GET/stats")
def get_prediction_stats() -> Dict[str, Any]:
    """Get prediction statistics.

    Returns counts of predictions, labeled/unlabeled, and accuracy.
    """
    from dags.utils import database as db
    
    return db.get_prediction_stats()
