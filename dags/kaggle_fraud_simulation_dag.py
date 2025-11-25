"""
Main DAG file for fraud detection data pipeline.

This DAG orchestrates:
1. Data partitioning (optional, runs periodically)
2. Daily data ingestion
3. Data drift detection
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from dags.config import DAG_START_DATE
from dags.tasks.database_setup import setup_prediction_database
from dags.tasks.data_preparation import prepare_partitions
from dags.tasks.data_ingestion import ingest_daily_slice
from dags.tasks.drift_detection import check_data_drift

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime.strptime(DAG_START_DATE, "%Y-%m-%d"),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "kaggle_fraud_simulation_daily",
    default_args=default_args,
    description="Daily fraud detection data ingestion with optional partitioning",
    schedule="@daily",
    catchup=True,  # Set to True if you want to backfill from start_date to today
    max_active_runs=1,
    tags=["fraud-detection", "kaggle", "data-ingestion"]
) as dag:
    
    # Task 0: Setup database (runs first, idempotent)
    task_setup_db = PythonOperator(
        task_id="setup_prediction_database",
        python_callable=setup_prediction_database,
    )
    
    # Task 1: Prepare partitioned data (runs once or periodically)
    task_prepare_partitions = PythonOperator(
        task_id="prepare_partitions",
        python_callable=prepare_partitions,
    )
    
    # Task 2: Daily ingestion (runs every day)
    task_ingest_daily = PythonOperator(
        task_id="ingest_daily_slice",
        python_callable=ingest_daily_slice,
    )
    
    # Task 3: Data drift detection (runs after ingestion)
    task_check_drift = PythonOperator(
        task_id="check_data_drift",
        python_callable=check_data_drift,
    )
    
    # Set task dependencies
    # Database Setup -> Partitioning -> Daily Ingestion -> Drift Detection
    task_setup_db >> task_prepare_partitions >> task_ingest_daily >> task_check_drift
