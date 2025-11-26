from fastapi import FastAPI, HTTPException
from typing import List, Optional, Dict, Any, Tuple
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
TRAINING_DATA_LIMIT = int(os.getenv("TRAINING_DATA_LIMIT", "20000"))


def _load_best_model_from_registry() -> Optional[Tuple[Any, Dict[str, Any], float]]:
    """Try to load the best Production model from MLflow Model Registry.
    
    Returns:
        Tuple of (model, artifacts, metric) if successful, None otherwise.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        import tempfile
        
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        model_name = "fraud-detection-xgboost"
        
        # Get latest Production model
        try:
            production_versions = client.get_latest_versions(
                model_name,
                stages=["Production"]
            )
            if not production_versions:
                logger.info("No Production model found in MLflow Model Registry")
                return None
            
            # Get the latest Production version
            latest_prod = production_versions[0]
            logger.info(f"Found Production model in registry: {model_name}/{latest_prod.version}")
            
            # Load model from MLflow
            model_uri = f"models:/{model_name}/Production"
            loaded_model = mlflow.sklearn.load_model(model_uri)
            
            # Load artifacts from the run
            run = client.get_run(latest_prod.run_id)
            artifacts = {}
            
            # Download artifacts from MLflow
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Download preprocessing artifacts
                try:
                    client.download_artifacts(
                        run_id=latest_prod.run_id,
                        path="preprocessing/preprocessing_artifacts.joblib",
                        dst_path=tmp_dir
                    )
                    artifacts_path = os.path.join(tmp_dir, "preprocessing_artifacts.joblib")
                    if os.path.exists(artifacts_path):
                        artifacts = joblib.load(artifacts_path)
                        logger.info("Loaded preprocessing artifacts from MLflow")
                except Exception as e:
                    logger.warning(f"Failed to load preprocessing artifacts from MLflow: {e}")
                    # Try to get train_cols from JSON
                    try:
                        client.download_artifacts(
                            run_id=latest_prod.run_id,
                            path="preprocessing/train_cols.json",
                            dst_path=tmp_dir
                        )
                        train_cols_path = os.path.join(tmp_dir, "train_cols.json")
                        if os.path.exists(train_cols_path):
                            with open(train_cols_path, "r") as f:
                                train_cols = json.load(f)
                            artifacts["train_cols"] = train_cols
                            logger.info("Loaded train_cols from MLflow")
                    except Exception as e2:
                        logger.warning(f"Failed to load train_cols from MLflow: {e2}")
            
            # Extract metrics
            metric = run.data.metrics.get("val_auc", 0.0)
            
            logger.info(f"Loaded Production model from MLflow (AUC: {metric:.4f})")
            return (loaded_model, artifacts, metric)
            
        except Exception as e:
            logger.warning(f"Failed to load model from MLflow registry: {e}")
            return None
            
    except ImportError:
        logger.info("MLflow not available, skipping registry model load")
        return None
    except Exception as e:
        logger.warning(f"Error accessing MLflow registry: {e}")
        return None


def _migrate_local_model_to_mlflow() -> bool:
    """Migrate existing local model to MLflow Model Registry.
    
    This is called on first startup if no models exist in MLflow registry.
    Loads the model from local files and registers it to MLflow.
    
    Returns:
        bool: True if migration successful, False otherwise
    """
    if not CANONICAL_MODEL_PATH.exists():
        logger.info("No local model found to migrate")
        return False
    
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        logger.info("Migrating local model to MLflow Model Registry...")
        
        # Load local model and artifacts
        local_model = joblib.load(CANONICAL_MODEL_PATH)
        local_artifacts = None
        if ARTIFACTS_PATH.exists():
            local_artifacts = joblib.load(ARTIFACTS_PATH)
        
        # Setup MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        
        # Get baseline metric from metadata
        meta = _load_model_meta()
        baseline_auc = meta.get("val_auc", 0.0) if meta else 0.0
        
        # Register model in MLflow
        model_name = "fraud-detection-xgboost"
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        
        with mlflow.start_run(run_name=f"migration-{timestamp}") as run:
            # Log metrics
            mlflow.log_metrics({"val_auc": baseline_auc, "val_accuracy": 0.0})
            
            # Log artifacts
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Save model
                tmp_model_path = os.path.join(tmp_dir, "xgb_model.joblib")
                joblib.dump(local_model, tmp_model_path)
                mlflow.log_artifact(tmp_model_path, artifact_path="model_artifacts")
                
                # Save preprocessing artifacts if available
                if local_artifacts:
                    tmp_artifacts_path = os.path.join(tmp_dir, "preprocessing_artifacts.joblib")
                    joblib.dump(local_artifacts, tmp_artifacts_path)
                    mlflow.log_artifact(tmp_artifacts_path, artifact_path="preprocessing")
                    
                    # Save train_cols as JSON if available
                    if local_artifacts.get("train_cols"):
                        tmp_cols_path = os.path.join(tmp_dir, "train_cols.json")
                        with open(tmp_cols_path, "w") as f:
                            json.dump(local_artifacts["train_cols"], f)
                        mlflow.log_artifact(tmp_cols_path, artifact_path="preprocessing")
            
            # Register model
            model_version = mlflow.sklearn.log_model(
                local_model,
                artifact_path="sklearn-model",
                registered_model_name=model_name
            )
            
            # Transition to Production
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            client.transition_model_version_stage(
                name=model_name,
                version=str(model_version.version),
                stage="Production"
            )
            
            logger.info(f"Successfully migrated local model to MLflow as {model_name}/{model_version.version}")
            return True
            
    except ImportError:
        logger.error("MLflow not available, cannot migrate model")
        return False
    except Exception as e:
        logger.exception(f"Failed to migrate local model to MLflow: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load artifacts and model from MLflow Model Registry.
    
    On first startup, migrates existing local models to MLflow if needed.
    """
    model = FraudDetectionModel()
    artifacts = None
    best_metric = None
    
    # Try to load best model from MLflow Model Registry
    registry_result = _load_best_model_from_registry()
    if registry_result:
        registry_model, registry_artifacts, registry_metric = registry_result
        model.model = registry_model
        artifacts = registry_artifacts
        best_metric = registry_metric
        logger.info("Loaded Production model from MLflow Model Registry")
    else:
        # No model in registry - try to migrate from local files
        logger.info("No model found in MLflow registry, attempting to migrate local model...")
        if _migrate_local_model_to_mlflow():
            # Retry loading from registry after migration
            registry_result = _load_best_model_from_registry()
            if registry_result:
                registry_model, registry_artifacts, registry_metric = registry_result
                model.model = registry_model
                artifacts = registry_artifacts
                best_metric = registry_metric
                logger.info("Loaded migrated model from MLflow Model Registry")
            else:
                logger.error("Failed to load model after migration")
        else:
            logger.warning("No model available in MLflow registry and migration failed")
    
    # If still no artifacts, try to load from local (for backward compatibility during migration)
    if not artifacts:
        artifacts = preprocessing.load_preprocessing_artifacts(ARTIFACTS_PATH.as_posix())
    
    # Get metric from metadata if not from registry
    if best_metric is None:
        meta = _load_model_meta()
        best_metric = meta.get("val_auc") if meta else None
    
    # Establish baseline if needed
    if best_metric is None and artifacts and model.model is not None:
        try:
            baseline = _evaluate_against_master(model, artifacts, BASELINE_SAMPLE_SIZE)
            if baseline is not None:
                best_metric = baseline
                _save_model_meta({
                    "val_auc": baseline, 
                    "run_id": "bootstrap",
                    "training_date": datetime.utcnow().strftime("%Y-%m-%d")
                })
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

        promoted = _maybe_promote_model(result)
        message = (
            f"New model promoted as serving model (trained on {len(training_df)} rows: {fraud_count} fraud, {non_fraud_count} non-fraud)"
            if promoted
            else f"Model logged to MLflow but not promoted (metric not improved). Trained on {len(training_df)} rows."
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

    # Transition model to Production stage in MLflow Model Registry (no local file saving)
    if result.model_version:
        try:
            _transition_model_to_production(result.model_version)
        except Exception as e:
            logger.warning(f"Failed to transition model to Production in MLflow: {e}")
            return False  # Fail if we can't register to MLflow
    else:
        logger.error("Model version not available, cannot promote to Production")
        return False

    # Load model from MLflow registry for serving
    registry_result = _load_best_model_from_registry()
    if not registry_result:
        logger.error("Failed to load promoted model from MLflow registry")
        return False
    
    registry_model, registry_artifacts, _ = registry_result
    
    # Save metadata (but not model files - models are in MLflow only)
    _save_model_meta(
        {
            "val_auc": candidate_metric,
            "run_id": result.run_id,
            "model_version": result.model_version,
            "updated_at": datetime.utcnow().isoformat(),
            "training_date": datetime.utcnow().strftime("%Y-%m-%d"),  # Store training date for drift detection
        }
    )
    
    app.state.best_metric = candidate_metric
    app.state.artifacts = registry_artifacts if registry_artifacts else result.artifacts
    app.state.model.model = registry_model
    logger.info("Promoted new model (AUC=%.4f) from run %s to MLflow Production", candidate_metric, result.run_id)
    return True


def _transition_model_to_production(model_version: str) -> None:
    """Transition a model version to Production stage in MLflow Model Registry.
    
    Also archives any previous Production models to Archived stage.
    
    Args:
        model_version: Model version string in format "model_name/version"
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        
        # Parse model name and version
        parts = model_version.split("/")
        if len(parts) != 2:
            logger.warning(f"Invalid model_version format: {model_version}")
            return
        
        model_name, version_str = parts
        
        # Archive any existing Production models
        try:
            production_versions = client.get_latest_versions(
                model_name, 
                stages=["Production"]
            )
            for pv in production_versions:
                if pv.version != version_str:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=pv.version,
                        stage="Archived"
                    )
                    logger.info(f"Archived previous Production model: {model_name}/{pv.version}")
        except Exception as e:
            logger.warning(f"Failed to archive previous Production models: {e}")
        
        # Transition new model to Production
        client.transition_model_version_stage(
            name=model_name,
            version=version_str,
            stage="Production"
        )
        logger.info(f"Transitioned model {model_version} to Production stage in MLflow Model Registry")
        
    except ImportError:
        logger.warning("MLflow not available, skipping model registry transition")
    except Exception as e:
        logger.warning(f"Failed to transition model to Production: {e}")


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
