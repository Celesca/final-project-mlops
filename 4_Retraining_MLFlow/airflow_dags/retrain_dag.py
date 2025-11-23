from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from retraining.retrain_pipeline import retrain_pipeline

with DAG(
    "retrain_pipeline",
    start_date=datetime(2024,1,1),
    schedule_interval=None
):
    run = PythonOperator(
        task_id="run_retraining",
        python_callable=retrain_pipeline
    )