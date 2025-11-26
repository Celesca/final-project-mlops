"""
Data drift detection tasks for monitoring data quality.
"""
import os
import glob
import pandas as pd
from datetime import datetime as dt
from dags.config import DATA_DIR, DRIFT_THRESHOLD
from dags.utils.drift_detector import is_drift_critical
from dags.utils.training_metadata import get_baseline_date, save_baseline_date, get_training_date


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


def check_data_drift(ds, **kwargs):
    """
    Checks for data drift by comparing the latest ingestion date's data with the training date's data
    (or baseline date if no training date exists).
    This detects changes between the reference data and the most recent ingested data.
    If no training date or baseline date exists, stores the current data as baseline for future comparisons.
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

    # Load latest ingestion data first (needed for baseline storage if no reference exists)
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

    # Get the training date or baseline date from metadata
    training_date = get_training_date()
    baseline_date = get_baseline_date() if not training_date else None
    
    # If no training date and no baseline date, store current data as baseline
    if not training_date and not baseline_date:
        print("ℹ️ No training date or baseline date found. Storing current data as baseline for future drift detection.")
        save_baseline_date(latest_ingestion_date)
        baseline_date = latest_ingestion_date
        print(f"✅ Baseline date set to: {baseline_date}")
        print("ℹ️ Drift detection will be available starting from the next run.")
        return
    
    # Use training date if available, otherwise use baseline date
    reference_date = training_date if training_date else baseline_date
    date_type = "training date" if training_date else "baseline date"
    
    print(f"📅 Reference {date_type}: {reference_date}")

    # If latest ingestion date is the same as reference date, no drift to check
    if latest_ingestion_date == reference_date:
        print(f"ℹ️ Latest ingestion date ({latest_ingestion_date}) matches {date_type}. No drift to check.")
        return

    # Load reference date data (reference/baseline for drift detection)
    reference_data_path = os.path.join(DATA_DIR, f"processed_fraud_data_{reference_date}.parquet")
    if not os.path.exists(reference_data_path):
        # Fallback to CSV
        reference_data_path = os.path.join(DATA_DIR, f"processed_fraud_data_{reference_date}.csv")
        if not os.path.exists(reference_data_path):
            print(f"⚠️ Reference {date_type} data file not found for {reference_date}. Cannot perform drift detection.")
            print("ℹ️ This might happen if the reference date file was deleted or not ingested yet.")
            return
        reference_data = pd.read_csv(reference_data_path)
    else:
        reference_data = pd.read_parquet(reference_data_path)

    if reference_data.empty:
        print(f"⚠️ Reference {date_type} data is empty, skipping drift detection.")
        return
    
    # Ensure both datasets have the same columns (select only common columns)
    common_columns = list(set(reference_data.columns) & set(current_data.columns))
    if not common_columns:
        print(f"⚠️ No common columns between {date_type} and latest ingestion data.")
        return

    reference_data = reference_data[common_columns]
    current_data = current_data[common_columns]

    print(f"📊 Comparing data: {date_type.capitalize()} ({reference_date}, {len(reference_data)} rows) vs Latest ingestion ({latest_ingestion_date}, {len(current_data)} rows)")
    print(f"📋 Columns being checked: {len(common_columns)} columns")
    
    # Check for drift
    has_drift = is_drift_critical(
        reference_data=reference_data,
        current_data=current_data,
        threshold=DRIFT_THRESHOLD
    )
    
    if has_drift:
        print(f"🚨 ALERT: Data drift detected! Drift exceeds threshold of {DRIFT_THRESHOLD*100}%")
        print(f"⚠️ Latest ingestion ({latest_ingestion_date}) shows significant drift compared to {date_type} ({reference_date})")
        print("💡 This could indicate:")
        print("   - Data distribution has changed since baseline/training")
        print("   - Model may need retraining")
        print("   - Potential data quality issues")
        print("   - Anomalous patterns requiring investigation")
        # In production, you might want to:
        # - Send alerts (email, Slack, etc.)
        # - Trigger model retraining
        # - Log to monitoring system
    else:
        print(f"✅ No significant drift detected. Data distribution is stable compared to {date_type}.")
    
    return has_drift

