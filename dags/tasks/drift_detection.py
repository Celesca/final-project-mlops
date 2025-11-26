"""
Data drift detection tasks for monitoring data quality.
"""
import os
import glob
import json
import pandas as pd
from datetime import datetime as dt
from pathlib import Path
from dags.config import DATA_DIR, DRIFT_THRESHOLD
from dags.utils.drift_detector import is_drift_critical


def _get_latest_ingestion_date() -> str:
    """
    Get the latest date from daily ingestion files.
    
    Returns:
        Date string in YYYY-MM-DD format, or None if no files found
    """
    # Look for processed_fraud_data_{date}.parquet files
    pattern = os.path.join(DATA_DIR, "processed_fraud_data_*.parquet")
    parquet_files = glob.glob(pattern)
    
    # Also check CSV files as fallback
    if not parquet_files:
        pattern = os.path.join(DATA_DIR, "processed_fraud_data_*.csv")
        csv_files = glob.glob(pattern)
        files = csv_files
    else:
        files = parquet_files
    
    if not files:
        return None
    
    # Extract dates from filenames
    dates = []
    for file in files:
        filename = os.path.basename(file)
        # Extract date from processed_fraud_data_{date}.parquet or .csv
        if filename.startswith("processed_fraud_data_") and (filename.endswith(".parquet") or filename.endswith(".csv")):
            date_str = filename.replace("processed_fraud_data_", "").replace(".parquet", "").replace(".csv", "")
            try:
                # Validate date format
                dt.strptime(date_str, "%Y-%m-%d")
                dates.append(date_str)
            except ValueError:
                continue
    
    if not dates:
        return None
    
    # Return the latest (max) date
    return max(dates)


def _get_training_date() -> str:
    """
    Get the training date from the metadata file.
    
    Returns:
        Date string in YYYY-MM-DD format, or None if not found
    """
    # Path to training metadata (in model serving models directory)
    # Try to find it relative to the dags directory
    repo_root = Path(__file__).resolve().parents[2]
    metadata_path = repo_root / "3_Model_Serving" / "models" / "training_metadata.json"
    
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata.get('last_training_date')
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ Error reading training metadata: {e}")
        return None


def check_data_drift(ds, **kwargs):
    """
    Checks for data drift by comparing the latest ingestion date's data with the training date's data.
    This detects changes between the data used for training and the most recent ingested data.
    Logs warnings if drift exceeds the threshold.
    
    Args:
        ds (str): The logical date of the run (YYYY-MM-DD), provided by Airflow.
    """
    
    print(f"🔍 Starting drift detection for logical date: {ds}")
    
    # Get the latest ingestion date from daily files
    latest_ingestion_date = _get_latest_ingestion_date()
    if not latest_ingestion_date:
        print("⚠️ No ingestion files found. Cannot determine latest ingestion date.")
        return
    
    print(f"📅 Latest ingestion date found: {latest_ingestion_date}")
    
    # Get the training date from metadata
    training_date = _get_training_date()
    if not training_date:
        print("⚠️ No training date found in metadata. Cannot perform drift detection.")
        print("ℹ️ This might happen if no model has been trained yet.")
        return
    
    print(f"📅 Training date: {training_date}")
    
    # If latest ingestion date is the same as training date, no drift to check
    if latest_ingestion_date == training_date:
        print(f"ℹ️ Latest ingestion date ({latest_ingestion_date}) matches training date. No drift to check.")
        return
    
    # Load latest ingestion data (current/reference for drift detection)
    latest_data_path = os.path.join(DATA_DIR, f"processed_fraud_data_{latest_ingestion_date}.parquet")
    if not os.path.exists(latest_data_path):
        # Fallback to CSV
        latest_data_path = os.path.join(DATA_DIR, f"processed_fraud_data_{latest_ingestion_date}.csv")
        if not os.path.exists(latest_data_path):
            print(f"⚠️ Latest ingestion data file not found for {latest_ingestion_date}, skipping drift detection.")
            return
        current_data = pd.read_csv(latest_data_path)
    else:
        current_data = pd.read_parquet(latest_data_path)
    
    if current_data.empty:
        print(f"⚠️ Latest ingestion data is empty for {latest_ingestion_date}, skipping drift detection.")
        return
    
    # Load training date data (reference/baseline for drift detection)
    training_data_path = os.path.join(DATA_DIR, f"processed_fraud_data_{training_date}.parquet")
    if not os.path.exists(training_data_path):
        # Fallback to CSV
        training_data_path = os.path.join(DATA_DIR, f"processed_fraud_data_{training_date}.csv")
        if not os.path.exists(training_data_path):
            print(f"⚠️ Training date data file not found for {training_date}. Cannot perform drift detection.")
            print("ℹ️ This might happen if the training date file was deleted or not ingested yet.")
            return
        reference_data = pd.read_csv(training_data_path)
    else:
        reference_data = pd.read_parquet(training_data_path)
    
    if reference_data.empty:
        print(f"⚠️ Training date data is empty, skipping drift detection.")
        return
    
    # Ensure both datasets have the same columns (select only common columns)
    common_columns = list(set(reference_data.columns) & set(current_data.columns))
    if not common_columns:
        print("⚠️ No common columns between training date and latest ingestion data.")
        return
    
    reference_data = reference_data[common_columns]
    current_data = current_data[common_columns]
    
    print(f"📊 Comparing data: Training date ({training_date}, {len(reference_data)} rows) vs Latest ingestion ({latest_ingestion_date}, {len(current_data)} rows)")
    print(f"📋 Columns being checked: {len(common_columns)} columns")
    
    # Check for drift
    has_drift = is_drift_critical(
        reference_data=reference_data,
        current_data=current_data,
        threshold=DRIFT_THRESHOLD
    )
    
    if has_drift:
        print(f"🚨 ALERT: Data drift detected! Drift exceeds threshold of {DRIFT_THRESHOLD*100}%")
        print(f"⚠️ Latest ingestion ({latest_ingestion_date}) shows significant drift compared to training date ({training_date})")
        print("💡 This could indicate:")
        print("   - Data distribution has changed since model training")
        print("   - Model may need retraining")
        print("   - Potential data quality issues")
        print("   - Anomalous patterns requiring investigation")
        # In production, you might want to:
        # - Send alerts (email, Slack, etc.)
        # - Trigger model retraining
        # - Log to monitoring system
    else:
        print(f"✅ No significant drift detected. Data distribution is stable compared to training date.")
    
    return has_drift

