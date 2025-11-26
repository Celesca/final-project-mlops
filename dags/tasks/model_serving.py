"""
Tasks that interact with the model serving API (train + predict) and persist results.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Any, Iterable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import pandas as pd
import requests

from dags.config import MODEL_SERVING_BASE_URL
from dags.utils.database import (
    save_prediction,
    save_predictions_bulk,
    save_transaction_record,
    save_transactions_bulk,
    build_transaction_key,
    get_transaction_lookup_for_ingest,
)

# Configuration for optimization
PREDICTION_CHUNK_SIZE = int(os.getenv("PREDICTION_CHUNK_SIZE", "1000"))  # Increased from 200
MAX_WORKERS = int(os.getenv("PREDICTION_MAX_WORKERS", "10"))  # Parallel API calls
BATCH_SIZE = int(os.getenv("PREDICTION_BATCH_SIZE", "5000"))  # Database batch size


def _chunk_records(records: List[Dict[str, Any]], chunk_size: int = PREDICTION_CHUNK_SIZE) -> Iterable[List[Dict[str, Any]]]:
    """Split records into chunks for batch processing."""
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


def _predict_batch(batch: List[Dict[str, Any]], predict_url: str) -> List[Dict[str, Any]]:
    """Make a single prediction API call for a batch of records."""
    try:
        payload = {"new_data": batch}
        resp = requests.post(predict_url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ Error predicting batch of {len(batch)} records: {e}")
        raise


def score_daily_predictions(ds, **context) -> Dict[str, Any]:
    """
    Optimized version: Sends the ingested daily slice to the model serving /predict endpoint
    using parallel API calls, then stores prediction outcomes in Postgres using bulk inserts.
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

    ingest_date = ingestion_output.get("ingest_date")
    if not ingest_date:
        raise ValueError("Ingest date missing; cannot map predictions to transactions.")

    print(f"📦 Processing {len(df)} records (from {source_used})")
    
    # Get transaction lookup once
    tx_lookup = get_transaction_lookup_for_ingest(ingest_date)
    
    # Convert to records
    records = df.to_dict(orient="records")
    predict_url = f"{MODEL_SERVING_BASE_URL.rstrip('/')}/predict"
    
    # Step 1: Parallel API calls for predictions
    print(f"🚀 Starting parallel prediction API calls (chunk_size={PREDICTION_CHUNK_SIZE}, workers={MAX_WORKERS})...")
    batches = list(_chunk_records(records, PREDICTION_CHUNK_SIZE))
    all_enriched = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all batches in parallel
        future_to_batch = {
            executor.submit(_predict_batch, batch, predict_url): i 
            for i, batch in enumerate(batches)
        }
        
        # Collect results as they complete
        results = [None] * len(batches)
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                enriched = future.result()
                results[batch_idx] = enriched
                print(f"✅ Completed batch {batch_idx + 1}/{len(batches)} ({len(enriched)} records)")
            except Exception as e:
                print(f"❌ Batch {batch_idx + 1} failed: {e}")
                raise
        
        # Flatten results in order
        for enriched in results:
            if enriched:
                all_enriched.extend(enriched)
    
    total_scored = len(all_enriched)
    print(f"✅ Completed all predictions: {total_scored} records scored")
    
    # Step 2: Prepare data for bulk database operations
    prediction_time = datetime.utcnow()
    transactions_to_save = []
    transactions_with_predictions = []  # Store (item, prediction, predict_proba) tuples
    predictions_to_save = []
    
    print(f"📝 Preparing {total_scored} records for database storage...")
    
    # First pass: identify missing transactions and prepare predictions for existing ones
    for item in all_enriched:
        prediction = bool(item.get("prediction", 0))
        predict_proba = float(item.get("predict_proba", 0.0))
        tx_key = build_transaction_key(item, ingest_date=ingest_date)
        transaction_id = tx_lookup.get(tx_key)
        
        if transaction_id is None:
            # Track for bulk insert along with prediction data
            transactions_to_save.append(item)
            transactions_with_predictions.append((item, prediction, predict_proba))
        else:
            # Ready to save prediction
            predictions_to_save.append((transaction_id, prediction, predict_proba, prediction_time))
    
    # Step 3: Bulk save missing transactions
    lookup_misses = 0
    if transactions_to_save:
        print(f"💾 Bulk saving {len(transactions_to_save)} missing transactions...")
        try:
            new_tx_ids = save_transactions_bulk(transactions_to_save)
            
            # Step 4: Add predictions for newly saved transactions
            for i, (item, prediction, predict_proba) in enumerate(transactions_with_predictions):
                transaction_id = new_tx_ids[i]
                tx_key = build_transaction_key(item, ingest_date=ingest_date)
                tx_lookup[tx_key] = transaction_id  # Update lookup for future use
                
                predictions_to_save.append((transaction_id, prediction, predict_proba, prediction_time))
            
            lookup_misses = len(new_tx_ids)
            print(f"✅ Saved {lookup_misses} new transactions")
        except Exception as e:
            print(f"❌ Error bulk saving transactions: {e}")
            raise
    
    # Step 5: Bulk save all predictions
    stored_predictions = 0
    if predictions_to_save:
        print(f"💾 Bulk saving {len(predictions_to_save)} predictions...")
        try:
            # Save in batches to avoid memory issues
            for i in range(0, len(predictions_to_save), BATCH_SIZE):
                batch = predictions_to_save[i:i + BATCH_SIZE]
                save_predictions_bulk(batch)
                stored_predictions += len(batch)
            print(f"✅ Saved {stored_predictions} predictions")
        except Exception as e:
            print(f"❌ Error bulk saving predictions: {e}")
            raise
    
    if lookup_misses:
        print(f"ℹ️ Added {lookup_misses} missing transactions while storing predictions.")
    print(f"✅ Scored {total_scored} rows; stored {stored_predictions} predictions (actual_label=NULL).")
    return {
        "scored_rows": total_scored,
        "stored_predictions": stored_predictions,
    }

