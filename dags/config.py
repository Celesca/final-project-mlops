"""
Configuration constants for the fraud detection DAG.
"""
import os

# Dataset configuration
DATASET_NAME = "sriharshaeedala/financial-fraud-detection-dataset"
CSV_FILENAME = "PS_20174392719_1491204439457_log.csv"

# Data paths
DATA_DIR = "/opt/airflow/data"
PARTITIONED_DATA_DIR = os.path.join(DATA_DIR, "partitioned_data")

# Database configuration (Postgres for master + prediction tables)
FRAUD_DB_CONFIG = {
    "host": os.getenv("FRAUD_DB_HOST", "fraud-db"),
    "port": int(os.getenv("FRAUD_DB_PORT", "5432")),
    "dbname": os.getenv("FRAUD_DB_NAME", "frauddb"),
    "user": os.getenv("FRAUD_DB_USER", "fraud"),
    "password": os.getenv("FRAUD_DB_PASSWORD", "fraud123"),
}

# Model serving API configuration
MODEL_SERVING_BASE_URL = os.getenv("MODEL_SERVING_BASE_URL", "http://host.docker.internal:8000")

# DAG configuration
DAG_START_DATE = "2025-10-23"  # The anchor date for simulation
MAX_SIMULATION_DAYS = 30  # Total number of days in the dataset (0-indexed: days 0-29)

# Drift detection configuration
DRIFT_THRESHOLD = 0.3  # 30% of columns can drift before alerting

