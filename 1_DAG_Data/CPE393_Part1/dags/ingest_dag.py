from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
import os

def ingest_data():
    input_path = "/opt/airflow/data/Synthetic_Financial_datasets_log.csv"

    df = pd.read_csv(input_path)

    output_path = os.path.join("/opt/airflow/data", "processed_data.csv")
    df.to_csv(output_path, index=False)

    print(f"Loaded input CSV: {input_path}")
    print(f"Processed data saved to: {output_path}")

default_args = {'start_date': datetime(2025, 11, 23)}

dag = DAG('data_ingestion', default_args=default_args, schedule_interval='@daily')

task = PythonOperator(
    task_id='ingest_data_task',
    python_callable=ingest_data,
    dag=dag
)
