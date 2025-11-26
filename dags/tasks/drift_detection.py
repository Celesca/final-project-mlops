"""
Data drift detection tasks for monitoring data quality.
"""
import os
import pandas as pd
from datetime import datetime as dt, timedelta
from dags.config import DATA_DIR, DAG_START_DATE, DRIFT_THRESHOLD
from dags.utils.drift_detector import is_drift_critical
from dags.utils.model_metadata import get_latest_training_date


def check_data_drift(ds, **kwargs):
    """
    Checks for data drift by comparing current day's data with the baseline data
    from the latest training date (or first date if no training date is available).
    This detects changes in data distribution since the model was last trained.
    Logs warnings if drift exceeds the threshold.
    
    Args:
        ds (str): The logical date of the run (YYYY-MM-DD), provided by Airflow.
    """
    
    print(f"🔍 Starting drift detection for logical date: {ds}")
    
    # Calculate simulation day
    dag_start = dt.strptime(DAG_START_DATE, "%Y-%m-%d")
    current_run = dt.strptime(ds, "%Y-%m-%d")
    delta_days = (current_run - dag_start).days
    
    if delta_days < 0:
        print("⚠️ Run date is before start date, skipping drift detection.")
        return
    
    # Get the baseline training date (latest training date or first date)
    baseline_date = get_latest_training_date()
    baseline_dt = dt.strptime(baseline_date, "%Y-%m-%d")
    
    # Skip if current date is before or equal to baseline date
    if current_run <= baseline_dt:
        print(f"ℹ️ Current date ({ds}) is on or before baseline training date ({baseline_date}). Skipping drift detection.")
        return
    
    # Load current day's data
    current_data_path = os.path.join(DATA_DIR, f"processed_fraud_data_{ds}.parquet")
    if not os.path.exists(current_data_path):
        # Fallback to CSV
        current_data_path = os.path.join(DATA_DIR, f"processed_fraud_data_{ds}.csv")
        if not os.path.exists(current_data_path):
            print(f"⚠️ No data file found for {ds}, skipping drift detection.")
            return
        current_data = pd.read_csv(current_data_path)
    else:
        current_data = pd.read_parquet(current_data_path)
    
    if current_data.empty:
        print(f"⚠️ Current data is empty for {ds}, skipping drift detection.")
        return
    
    # Load baseline/reference data from training date
    baseline_path = os.path.join(DATA_DIR, f"processed_fraud_data_{baseline_date}.parquet")
    
    if not os.path.exists(baseline_path):
        # Fallback to CSV
        baseline_path = os.path.join(DATA_DIR, f"processed_fraud_data_{baseline_date}.csv")
        if not os.path.exists(baseline_path):
            print(f"⚠️ Baseline data ({baseline_date}) not found. Cannot perform drift detection.")
            print(f"ℹ️ This might happen if the training date data is not available.")
            return
        reference_data = pd.read_csv(baseline_path)
    else:
        reference_data = pd.read_parquet(baseline_path)
    
    if reference_data.empty:
        print(f"⚠️ Baseline data is empty, skipping drift detection.")
        return
    
    # Ensure both datasets have the same columns (select only common columns)
    common_columns = list(set(reference_data.columns) & set(current_data.columns))
    if not common_columns:
        print("⚠️ No common columns between baseline and current data.")
        return
    
    reference_data = reference_data[common_columns]
    current_data = current_data[common_columns]
    
    print(f"📊 Comparing data: Baseline/Training Date ({baseline_date}, {len(reference_data)} rows) vs Today ({ds}, {len(current_data)} rows)")
    print(f"📋 Columns being checked: {len(common_columns)} columns")
    
    # Check for drift
    has_drift = is_drift_critical(
        reference_data=reference_data,
        current_data=current_data,
        threshold=DRIFT_THRESHOLD
    )
    
    if has_drift:
        print(f"🚨 ALERT: Data drift detected! Drift exceeds threshold of {DRIFT_THRESHOLD*100}%")
        print(f"⚠️ Today ({ds}) shows significant drift compared to training baseline ({baseline_date})")
        print("💡 This could indicate:")
        print("   - Data distribution has changed since model training")
        print("   - Potential data quality issues")
        print("   - Model may need retraining")
        print("   - Anomalous patterns requiring investigation")
        # In production, you might want to:
        # - Send alerts (email, Slack, etc.)
        # - Trigger model retraining
        # - Log to monitoring system
    else:
        print(f"✅ No significant drift detected. Data distribution is stable compared to training baseline ({baseline_date}).")
    
    return has_drift

