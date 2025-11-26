"""
Utility functions to retrieve model metadata, particularly training dates.
"""
import os
import json
from typing import Optional
from datetime import datetime as dt

from dags.config import DAG_START_DATE, MODEL_SERVING_BASE_URL


def get_latest_training_date() -> str:
    """
    Get the latest training date from model metadata.
    
    Tries multiple sources in order:
    1. Model metadata file (best_model_meta.json) if accessible
    2. MLflow Model Registry (if available)
    3. Falls back to DAG_START_DATE
    
    Returns:
        str: Training date in YYYY-MM-DD format
    """
    # Try to read from model metadata file
    # Check if we can access the model serving models directory
    possible_paths = [
        "/opt/airflow/models/best_model_meta.json",  # If mounted in Airflow (docker-compose)
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                     "3_Model_Serving", "models", "best_model_meta.json"),  # Relative path (local dev)
        "/app/models/best_model_meta.json",  # Alternative mount point
    ]
    
    for meta_path in possible_paths:
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    training_date = meta.get("training_date")
                    if training_date:
                        # Validate date format
                        try:
                            dt.strptime(training_date, "%Y-%m-%d")
                            print(f"📅 Found training date from metadata: {training_date}")
                            return training_date
                        except ValueError:
                            print(f"⚠️ Invalid date format in metadata: {training_date}")
            except Exception as e:
                print(f"⚠️ Failed to read metadata from {meta_path}: {e}")
    
    # Try to get from MLflow Model Registry
    try:
        training_date = _get_training_date_from_mlflow()
        if training_date:
            return training_date
    except Exception as e:
        print(f"⚠️ Failed to get training date from MLflow: {e}")
    
    # Fall back to DAG_START_DATE
    print(f"ℹ️ No training date found, using default: {DAG_START_DATE}")
    return DAG_START_DATE


def _get_training_date_from_mlflow() -> Optional[str]:
    """
    Try to get training date from MLflow Model Registry.
    
    Returns:
        Optional[str]: Training date in YYYY-MM-DD format, or None if not available
    """
    try:
        from mlflow.tracking import MlflowClient
        
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
        client = MlflowClient(tracking_uri=mlflow_uri)
        model_name = "fraud-detection-xgboost"
        
        # Get latest Production model
        production_versions = client.get_latest_versions(
            model_name,
            stages=["Production"]
        )
        
        if production_versions:
            latest_prod = production_versions[0]
            run = client.get_run(latest_prod.run_id)
            
            # Extract start time and convert to date
            if run.info.start_time:
                # MLflow timestamps are in milliseconds
                start_time = run.info.start_time / 1000  # Convert to seconds
                training_date = dt.fromtimestamp(start_time).strftime("%Y-%m-%d")
                print(f"📅 Found training date from MLflow: {training_date}")
                return training_date
                
    except ImportError:
        print("ℹ️ MLflow not available")
    except Exception as e:
        print(f"⚠️ Error accessing MLflow: {e}")
    
    return None

