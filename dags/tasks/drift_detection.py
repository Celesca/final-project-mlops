"""
Data drift detection tasks for monitoring data quality.
"""
import os
import pandas as pd
from datetime import datetime as dt, timedelta
from dags.config import DATA_DIR, DAG_START_DATE, DRIFT_THRESHOLD
from dags.utils.drift_detector import is_drift_critical


def check_data_drift(ds, **kwargs):
    """
    Checks for data drift by comparing current day's data with yesterday's data.
    This detects sudden changes and anomalies in daily data patterns.
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
    
    simulation_day = delta_days
    
    # On the first day, there's no yesterday to compare, so skip drift detection
    if simulation_day == 0:
        print("ℹ️ This is the first day, no previous day to compare. Skipping drift detection.")
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
    
    # Calculate yesterday's date
    yesterday_date = (current_run - timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_path = os.path.join(DATA_DIR, f"processed_fraud_data_{yesterday_date}.parquet")
    
    if not os.path.exists(yesterday_path):
        # Fallback to CSV
        yesterday_path = os.path.join(DATA_DIR, f"processed_fraud_data_{yesterday_date}.csv")
        if not os.path.exists(yesterday_path):
            print(f"⚠️ Yesterday's data ({yesterday_date}) not found. Cannot perform drift detection.")
            print("ℹ️ This might happen on the first run or if yesterday's ingestion failed.")
            return
        reference_data = pd.read_csv(yesterday_path)
    else:
        reference_data = pd.read_parquet(yesterday_path)
    
    if reference_data.empty:
        print(f"⚠️ Yesterday's data is empty, skipping drift detection.")
        return
    
    # Ensure both datasets have the same columns (select only common columns)
    common_columns = list(set(reference_data.columns) & set(current_data.columns))
    if not common_columns:
        print("⚠️ No common columns between yesterday's and current data.")
        return
    
    reference_data = reference_data[common_columns]
    current_data = current_data[common_columns]
    
    print(f"📊 Comparing data: Yesterday ({yesterday_date}, {len(reference_data)} rows) vs Today ({ds}, {len(current_data)} rows)")
    print(f"📋 Columns being checked: {len(common_columns)} columns")
    
    # Check for drift
    has_drift = is_drift_critical(
        reference_data=reference_data,
        current_data=current_data,
        threshold=DRIFT_THRESHOLD
    )
    
    if has_drift:
        print(f"🚨 ALERT: Data drift detected! Drift exceeds threshold of {DRIFT_THRESHOLD*100}%")
        print(f"⚠️ Today ({ds}) shows significant drift compared to yesterday ({yesterday_date})")
        print("💡 This could indicate:")
        print("   - Sudden change in data distribution")
        print("   - Potential data quality issues")
        print("   - Anomalous patterns requiring investigation")
        # In production, you might want to:
        # - Send alerts (email, Slack, etc.)
        # - Trigger model retraining
        # - Log to monitoring system
    else:
        print(f"✅ No significant drift detected. Data distribution is stable compared to yesterday.")
    
    return has_drift

