"""
Training utilities that build and log a fraud detection model via MLflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import tempfile
from typing import Dict, Any

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from . import preprocessing


@dataclass
class TrainingResult:
    model: Any
    artifacts: Dict[str, Any]
    metrics: Dict[str, float]
    run_id: str
    train_rows: int
    val_rows: int
    feature_count: int


def train_new_model(
    dataset: pd.DataFrame,
    tracking_uri: str,
    experiment_name: str,
    random_state: int = 42,
    test_size: float = 0.2,
) -> TrainingResult:
    """
    Train a new XGBoost model from the provided dataset and log to MLflow.
    """
    if dataset.empty:
        raise ValueError("Training dataset is empty")

    if "isFraud" not in dataset.columns:
        raise ValueError("Training dataset must include 'isFraud' column")

    labelled_df = dataset.dropna(subset=["isFraud"]).copy()
    if labelled_df.empty:
        raise ValueError("Training dataset has no rows with 'isFraud' labels")

    labelled_df["isFraud"] = labelled_df["isFraud"].astype(int)
    feature_source = preprocessing.build_feature_source_from_master(labelled_df)
    features, artifacts = preprocessing.prepare_feature_frame(feature_source, fit=True)
    target = labelled_df["isFraud"]

    if features.empty:
        raise ValueError("No usable features were produced from the dataset")

    stratify = target if target.nunique() > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

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
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    val_auc = roc_auc_score(y_val, val_probs) if y_val.nunique() > 1 else 0.0
    val_acc = accuracy_score(y_val, val_preds)

    metrics = {
        "val_auc": float(val_auc),
        "val_accuracy": float(val_acc),
    }

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"fraud-train-{timestamp}") as run:
        mlflow.log_params(
            {
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "learning_rate": model.learning_rate,
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "feature_count": features.shape[1],
            }
        )
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_model_path = os.path.join(tmp_dir, "model.joblib")
            joblib.dump(model, tmp_model_path)
            mlflow.log_artifact(tmp_model_path, artifact_path="model_artifacts")

            tmp_artifacts_path = os.path.join(tmp_dir, "preprocessing_artifacts.joblib")
            joblib.dump(artifacts, tmp_artifacts_path)
            mlflow.log_artifact(tmp_artifacts_path, artifact_path="preprocessing")

        mlflow.sklearn.log_model(model, artifact_path="sklearn-model")

        return TrainingResult(
            model=model,
            artifacts=artifacts,
            metrics=metrics,
            run_id=run.info.run_id,
            train_rows=len(X_train),
            val_rows=len(X_val),
            feature_count=features.shape[1],
        )

