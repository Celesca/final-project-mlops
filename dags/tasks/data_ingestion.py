"""
Data ingestion tasks for daily data processing.
"""
import os
import pandas as pd
from dags.config import DATA_DIR, PARTITIONED_DATA_DIR, MAX_SIMULATION_DAYS
from dags.utils.dataset_utils import get_dataset_path
from dags.utils.database import save_transactions_bulk
from dags.utils.training_metadata import get_next_simulation_day, get_simulation_date


def ingest_daily_slice(ds, **kwargs):
    """
    Simulates daily ingestion by slicing data based on the simulation day counter.
    First tries to use partitioned parquet files for efficiency, falls back to CSV if needed.
    Uses an incrementing simulation day counter to allow proper simulation regardless of execution date.
    
    Args:
        ds (str): The logical date of the run (YYYY-MM-DD), provided by Airflow (not used for simulation).
    """
    print(f"🚀 Starting ingestion for logical date: {ds}")
    
    # Get next simulation day (increments automatically)
    simulation_day = get_next_simulation_day()
    
    # Check if we've exceeded the available simulation days
    if simulation_day >= MAX_SIMULATION_DAYS:
        print(f"⚠️ Simulation day {simulation_day} exceeds maximum available days ({MAX_SIMULATION_DAYS}).")
        print(f"ℹ️ All {MAX_SIMULATION_DAYS} days of data have been processed. Simulation complete.")
        print("💡 To restart the simulation, reset the simulation day counter in the metadata file.")
        return {
            "record_count": 0,
            "simulation_complete": True,
            "message": f"All {MAX_SIMULATION_DAYS} days processed"
        }
    
    # Generate simulated date for file naming and downstream tasks
    simulated_date = get_simulation_date(simulation_day)
    print(f"📊 Processing simulation day: {simulation_day} (simulated date: {simulated_date})")
    print(f"📅 Progress: Day {simulation_day + 1} of {MAX_SIMULATION_DAYS}")
    
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
        start_step = (simulation_day * 24) + 1
        end_step = (simulation_day + 1) * 24
        
        print(f"🔍 Filtering steps {start_step} to {end_step}")
        
        # Read and filter
        df = pd.read_csv(csv_path)
        daily_df = df[(df['step'] >= start_step) & (df['step'] <= end_step)]
    
    if daily_df.empty:
        print(f"⚠️ No data found for simulation day {simulation_day}.")
        return
    
    # Save output (both CSV and optionally parquet for consistency)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Use simulated date for file naming (not the execution date)
    csv_output_path = os.path.join(DATA_DIR, f"processed_fraud_data_{simulated_date}.csv")
    daily_df.to_csv(csv_output_path, index=False)
    print(f"✅ Saved {len(daily_df)} rows to: {csv_output_path}")
    
    # Also save as parquet for better performance in downstream tasks
    parquet_output_path = os.path.join(DATA_DIR, f"processed_fraud_data_{simulated_date}.parquet")
    daily_df.to_parquet(parquet_output_path, compression='snappy', index=False)
    print(f"✅ Saved {len(daily_df)} rows to: {parquet_output_path}")
    
    # Persist raw transactions into database for downstream analytics
    try:
        transactions_payload = daily_df.to_dict(orient="records")
        save_transactions_bulk(
            transactions=transactions_payload,
            ingest_date=simulated_date,  # Use simulated date, not execution date
            source_file=source_file,
        )
        print(f"🗄️ Stored {len(transactions_payload)} transactions for {simulated_date} in the DB.")
    except Exception as exc:
        print(f"⚠️ Failed to store transactions in DB: {exc}")

    return {
        "record_count": len(daily_df),
        "csv_path": csv_output_path,
        "parquet_path": parquet_output_path,
        "ingest_date": simulated_date,  # Return simulated date for downstream tasks
        "simulation_day": simulation_day,
    }

