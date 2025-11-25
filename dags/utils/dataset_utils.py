"""
Dataset utility functions for downloading and accessing Kaggle datasets.
"""
import os
import kagglehub
from dags.config import DATASET_NAME, CSV_FILENAME


def get_dataset_path():
    """
    Downloads and returns the path to the Kaggle dataset CSV file.
    Handles filename variations gracefully.
    
    Returns:
        str: Path to the CSV file
    """
    path = kagglehub.dataset_download(DATASET_NAME)
    csv_path = os.path.join(path, CSV_FILENAME)
    
    # Fallback for filename changes
    if not os.path.exists(csv_path):
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV found in downloaded path")
        csv_path = os.path.join(path, csv_files[0])
    
    return csv_path

