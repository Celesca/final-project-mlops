"""
Data preparation tasks for partitioning datasets.
"""
import os
import pandas as pd
import shutil
from dags.config import PARTITIONED_DATA_DIR
from dags.utils.dataset_utils import get_dataset_path


def prepare_partitions(**kwargs):
    """
    Prepares partitioned parquet files from the full dataset.
    This is a one-time setup task that can be run periodically to refresh partitions.
    Partitions data by simulation_day for efficient daily access.
    """
    print("⬇️ Downloading dataset...")
    csv_path = get_dataset_path()
    
    print("📖 Reading CSV (This may take a moment)...")
    # Use pyarrow engine for faster reading if available
    try:
        df = pd.read_csv(csv_path, engine='pyarrow')
    except Exception:
        df = pd.read_csv(csv_path)
    
    print("🔪 Slicing into daily partitions...")
    # Logic: Step 1-24 = Day 0, 25-48 = Day 1, etc.
    df['simulation_day'] = (df['step'] - 1) // 24
    
    # Clear old partitioned data if exists
    if os.path.exists(PARTITIONED_DATA_DIR):
        print("🗑️ Clearing existing partitioned data...")
        shutil.rmtree(PARTITIONED_DATA_DIR)
    
    os.makedirs(PARTITIONED_DATA_DIR, exist_ok=True)
    
    # Write to Hive-style partitions: /data/partitioned_data/simulation_day=0/data.parquet
    print("💾 Writing partitioned parquet files...")
    df.to_parquet(
        PARTITIONED_DATA_DIR,
        partition_cols=['simulation_day'],
        compression='snappy'
    )
    
    print(f"✅ DONE! Data partitioned in {PARTITIONED_DATA_DIR}")
    print(f"📊 Total rows: {len(df)}, Days: {df['simulation_day'].nunique()}")

