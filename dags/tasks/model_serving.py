"""
Tasks that interact with the model serving API (train + predict) and persist results.
"""
from __future__ import annotations

import os
from typing import Dict, List, Any, Iterable

import pandas as pd
import requests

from dags.config import MODEL_SERVING_BASE_URL
from dags.utils.database import save_prediction


def _chunk_records(records: List[Dict[str, Any]], chunk_size: int = 200) -> Iterable[List[Dict[str, Any]]]:
    for idx in range(0, len(records), chunk_size):
        yield records[idx : idx + chunk_size]


def trigger_model_retrain(**context) -> Dict[str, Any]:
    """
    Calls the model serving /train endpoint if drift was detected.
    """
    ti = context["ti"]
    drift_detected = ti.xcom_pull(task_ids="check_data_drift")
    if not drift_detected:
        print("ℹ️ No significant drift detected; skipping retrain call.")
        return {"retrain_triggered": False}

    url = f"{MODEL_SERVING_BASE_URL.rstrip('/')}/train"
    print(f"🚀 Drift detected. Calling model serving retrain endpoint: {url}")
    response = requests.post(url, timeout=120)
    response.raise_for_status()
    data = response.json()
    print(f"✅ Retrain completed. Response: {data}")
    return {"retrain_triggered": True, "train_response": data}


def score_daily_predictions(ds, **context) -> Dict[str, Any]:
    """
    Sends the ingested daily slice to the model serving /predict endpoint,
    then stores prediction outcomes in Postgres.
    """
    ti = context["ti"]
    ingestion_output = ti.xcom_pull(task_ids="ingest_daily_slice")
    if not ingestion_output:
        raise ValueError("Ingestion output missing; cannot score daily data.")

    parquet_path = ingestion_output.get("parquet_path")
    csv_path = ingestion_output.get("csv_path")

    if parquet_path and os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        source_used = parquet_path
    elif csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        source_used = csv_path
    else:
        raise FileNotFoundError("Neither parquet nor csv file found for scoring task.")

    if df.empty:
        print("⚠️ Daily dataframe is empty; skipping prediction step.")
        return {"scored_rows": 0}

    records = df.to_dict(orient="records")
    print(f"📦 Sending {len(records)} records (from {source_used}) to /predict")

    predict_url = f"{MODEL_SERVING_BASE_URL.rstrip('/')}/predict"
    total_scored = 0
    stored_predictions = 0

    for batch in _chunk_records(records):
        payload = {"new_data": batch}
        resp = requests.post(predict_url, json=payload, timeout=120)
        resp.raise_for_status()
        enriched = resp.json()
        total_scored += len(enriched)

        for item in enriched:
            actual_label = item.get("isFraud")
            if actual_label is None:
                continue  # Can't score correctness without label
            prediction = bool(item.get("prediction", 0))
            predict_proba = float(item.get("predict_proba", 0.0))
            save_prediction(
                transaction=item,
                prediction=prediction,
                actual_label=bool(actual_label),
                predict_proba=predict_proba,
            )
            stored_predictions += 1

    print(f"✅ Scored {total_scored} rows; stored {stored_predictions} labelled predictions.")
    return {
        "scored_rows": total_scored,
        "stored_predictions": stored_predictions,
    }

