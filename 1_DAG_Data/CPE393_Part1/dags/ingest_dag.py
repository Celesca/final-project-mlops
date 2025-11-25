from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import os
import kagglehub

def ingest_from_kaggle():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download(
        "sriharshaeedala/financial-fraud-detection-dataset"
    )
    print(f"Dataset downloaded to: {path}")

    # หาไฟล์ csv
    csv_file = None
    for file in os.listdir(path):
        if file.endswith(".csv"):
            csv_file = os.path.join(path, file)
            break
    if csv_file is None:
        raise FileNotFoundError("No CSV file found in Kaggle dataset.")

    print(f"Reading CSV: {csv_file}")
    df = pd.read_csv(csv_file)

    # เซฟ CSV เป็นรายวัน
    output_path = f"/opt/airflow/data/processed_data_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")

default_args = {
    "start_date": datetime(2025, 11, 23)
}

dag = DAG(
    "kaggle_ingestion",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False
)

task_ingest = PythonOperator(
    task_id="ingest_kaggle_data",
    python_callable=ingest_from_kaggle,
    dag=dag
)
