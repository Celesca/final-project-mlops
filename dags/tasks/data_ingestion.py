"""
Data ingestion tasks for daily data processing.
"""
import os
import pandas as pd
from datetime import datetime as dt
from dags.config import DATA_DIR, PARTITIONED_DATA_DIR, DAG_START_DATE
from dags.utils.dataset_utils import get_dataset_path
from dags.utils.database import save_transactions_bulk


def ingest_daily_slice(ds, **kwargs):
    """
    Simulates daily ingestion by slicing data based on the DAG's logical date.
    First tries to use partitioned parquet files for efficiency, falls back to CSV if needed.
    
    Args:
        ds (str): The logical date of the run (YYYY-MM-DD), provided by Airflow.
    """
    print(f"🚀 Starting ingestion for logical date: {ds}")
    
    # Calculate simulation day
    dag_start = dt.strptime(DAG_START_DATE, "%Y-%m-%d")
    current_run = dt.strptime(ds, "%Y-%m-%d")
    delta_days = (current_run - dag_start).days
    
    if delta_days < 0:
        print("⚠️ Run date is before start date, skipping processing.")
        return
    
    simulation_day = delta_days
    print(f"📊 Processing simulation day: {simulation_day}")
    
    # Try to use partitioned data first (more efficient)
    partition_path = os.path.join(PARTITIONED_DATA_DIR, f"simulation_day={simulation_day}")
    
    source_file = None

    if os.path.exists(partition_path):
        print(f"📂 Reading from partitioned data: {partition_path}")
        daily_df = pd.read_parquet(partition_path)
        source_file = partition_path
    else:
        print("⚠️ Partitioned data not found, falling back to CSV...")
        csv_path = get_dataset_path()
        print(f"📂 Reading from CSV: {csv_path}")
        source_file = csv_path
        
        # Calculate step range: 1 Day = 24 Steps
        start_step = (delta_days * 24) + 1
        end_step = (delta_days + 1) * 24
        
        print(f"🔍 Filtering steps {start_step} to {end_step}")
        
        # Read and filter
        df = pd.read_csv(csv_path)
        daily_df = df[(df['step'] >= start_step) & (df['step'] <= end_step)]
    
    if daily_df.empty:
        print(f"⚠️ No data found for simulation day {simulation_day}.")
        return
    
    # Save output (both CSV and optionally parquet for consistency)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save as CSV (for backward compatibility)
    csv_output_path = os.path.join(DATA_DIR, f"processed_fraud_data_{ds}.csv")
    daily_df.to_csv(csv_output_path, index=False)
    print(f"✅ Saved {len(daily_df)} rows to: {csv_output_path}")
    
    # Also save as parquet for better performance in downstream tasks
    parquet_output_path = os.path.join(DATA_DIR, f"processed_fraud_data_{ds}.parquet")
    daily_df.to_parquet(parquet_output_path, compression='snappy', index=False)
    print(f"✅ Saved {len(daily_df)} rows to: {parquet_output_path}")
    
    # Persist raw transactions into database for downstream analytics
    try:
        transactions_payload = daily_df.to_dict(orient="records")
        save_transactions_bulk(
            transactions=transactions_payload,
            ingest_date=ds,
            source_file=source_file,
        )
        print(f"🗄️ Stored {len(transactions_payload)} transactions for {ds} in the DB.")
    except Exception as exc:
        print(f"⚠️ Failed to store transactions in DB: {exc}")

    return daily_df

