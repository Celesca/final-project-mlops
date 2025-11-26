"""
Training utilities that build and log a fraud detection model via MLflow.

References the training approach from MLOps_Train.ipynb:
- Uses XGBClassifier with SMOTE for handling imbalanced data
- Features: type (one-hot encoded), amount, balance columns
- Logs model and artifacts to MLflow for versioning
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import tempfile
from typing import Dict, Any, Optional
import logging

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from . import preprocessing

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model: Any
    artifacts: Dict[str, Any]
    metrics: Dict[str, float]
    run_id: str
    model_version: Optional[str]  # Format: "model_name/version"
    train_rows: int
    val_rows: int
    feature_count: int


def _setup_mlflow(tracking_uri: str, experiment_name: str) -> bool:
    """Setup MLflow tracking. Returns True if successful, False otherwise."""
    try:
        import mlflow
        
        # Try to connect to MLflow server
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        return True
    except Exception as e:
        logger.warning(f"Failed to connect to MLflow at {tracking_uri}: {e}")
        return False


def _log_to_mlflow(
    model: Any,
    artifacts: Dict[str, Any],
    metrics: Dict[str, float],
    train_rows: int,
    val_rows: int,
    feature_count: int,
) -> tuple[str, Optional[str]]:
    """Log model and metrics to MLflow. Returns (run_id, model_version)."""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    
    with mlflow.start_run(run_name=f"fraud-train-{timestamp}") as run:
        # Log hyperparameters
        mlflow.log_params({
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "subsample": model.subsample,
            "colsample_bytree": model.colsample_bytree,
            "train_rows": train_rows,
            "val_rows": val_rows,
            "feature_count": feature_count,
        })
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log artifacts
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save model
            tmp_model_path = os.path.join(tmp_dir, "xgb_model.joblib")
            joblib.dump(model, tmp_model_path)
            mlflow.log_artifact(tmp_model_path, artifact_path="model_artifacts")
            
            # Save preprocessing artifacts
            tmp_artifacts_path = os.path.join(tmp_dir, "preprocessing_artifacts.joblib")
            joblib.dump(artifacts, tmp_artifacts_path)
            mlflow.log_artifact(tmp_artifacts_path, artifact_path="preprocessing")
            
            # Save train columns as JSON
            if artifacts.get("train_cols"):
                import json
                tmp_cols_path = os.path.join(tmp_dir, "train_cols.json")
                with open(tmp_cols_path, "w") as f:
                    json.dump(artifacts["train_cols"], f)
                mlflow.log_artifact(tmp_cols_path, artifact_path="preprocessing")
        
        # Log sklearn model for easy loading and register in Model Registry
        model_version = mlflow.sklearn.log_model(
            model, 
            artifact_path="sklearn-model",
            registered_model_name="fraud-detection-xgboost"
        )
        
        model_version_str = f"{model_version.name}/{model_version.version}" if model_version else None
        logger.info(f"Logged run {run.info.run_id} to MLflow and registered model version {model_version_str}")
        return run.info.run_id, model_version_str


def train_new_model(
    dataset: pd.DataFrame,
    tracking_uri: str,
    experiment_name: str,
    random_state: int = 42,
    test_size: float = 0.2,
) -> TrainingResult:
    """
    Train a new XGBoost model from the provided dataset and log to MLflow.
    
    This follows the approach from MLOps_Train.ipynb:
    1. Feature selection: type, amount, balance columns
    2. One-hot encode transaction type
    3. Scale numerical features
    4. Handle imbalance with SMOTE (if imblearn available)
    5. Train XGBClassifier
    6. Log to MLflow for versioning
    """
    if dataset.empty:
        raise ValueError("Training dataset is empty")

    if "isFraud" not in dataset.columns:
        raise ValueError("Training dataset must include 'isFraud' column")

    # Filter to labeled data
    labelled_df = dataset.dropna(subset=["isFraud"]).copy()
    if labelled_df.empty:
        raise ValueError("Training dataset has no rows with 'isFraud' labels")

    labelled_df["isFraud"] = labelled_df["isFraud"].astype(int)
    
    # Build features using preprocessing module
    feature_source = preprocessing.build_feature_source_from_master(labelled_df)
    features, artifacts = preprocessing.prepare_feature_frame(feature_source, fit=True)
    target = labelled_df["isFraud"]

    if features.empty:
        raise ValueError("No usable features were produced from the dataset")

    logger.info(f"Training with {len(features)} samples, {features.shape[1]} features")
    logger.info(f"Class distribution: {target.value_counts().to_dict()}")

    # Train-test split with stratification
    stratify = target if target.nunique() > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Try to apply SMOTE for handling imbalanced data (as in MLOps_Train.ipynb)
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"Applied SMOTE: {len(X_train)} -> {len(X_train_resampled)} samples")
        X_train, y_train = X_train_resampled, y_train_resampled
    except ImportError:
        logger.warning("imblearn not available, training without SMOTE")

    # XGBoost model (similar to MLOps_Train.ipynb)
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",
        random_state=random_state,
    )
    
    logger.info("Training XGBoost model...")
    model.fit(X_train, y_train)

    # Evaluate on validation set
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    val_auc = roc_auc_score(y_val, val_probs) if y_val.nunique() > 1 else 0.0
    val_acc = accuracy_score(y_val, val_preds)

    metrics = {
        "val_auc": float(val_auc),
        "val_accuracy": float(val_acc),
    }
    
    logger.info(f"Validation AUC: {val_auc:.4f}, Accuracy: {val_acc:.4f}")

    # Try to log to MLflow
    run_id = "local"
    model_version = None
    mlflow_success = _setup_mlflow(tracking_uri, experiment_name)
    
    if mlflow_success:
        try:
            run_id, model_version = _log_to_mlflow(
                model=model,
                artifacts=artifacts,
                metrics=metrics,
                train_rows=len(X_train),
                val_rows=len(X_val),
                feature_count=features.shape[1],
            )
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
            run_id = f"local-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    else:
        run_id = f"local-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        logger.info(f"Training completed without MLflow. Run ID: {run_id}")

    return TrainingResult(
        model=model,
        artifacts=artifacts,
        metrics=metrics,
        run_id=run_id,
        model_version=model_version,
        train_rows=len(X_train),
        val_rows=len(X_val),
        feature_count=features.shape[1],
    )

