"""
Utilities for interacting with the master transaction store (Postgres).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional

import pandas as pd
import psycopg2


DB_CONFIG = {
    "host": os.getenv("FRAUD_DB_HOST", "fraud-db"),
    "port": int(os.getenv("FRAUD_DB_PORT", "5432")),
    "dbname": os.getenv("FRAUD_DB_NAME", "frauddb"),
    "user": os.getenv("FRAUD_DB_USER", "fraud"),
    "password": os.getenv("FRAUD_DB_PASSWORD", "fraud123"),
}


@contextmanager
def get_connection():
    conn = psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
    )
    try:
        yield conn
    finally:
        conn.close()


def fetch_master_transactions(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch rows from the master `all_transactions` table.

    Args:
        limit: Maximum rows to pull (None fetches all rows).
    """
    base_query = """
        SELECT step, type, amount, "nameOrig", "oldbalanceOrg", "newbalanceOrig",
               "nameDest", "oldbalanceDest", "newbalanceDest",
               "isFraud", "isFlaggedFraud", ingest_date, source_file, created_at
        FROM all_transactions
    """
    order_clause = " ORDER BY created_at DESC"
    limit_clause = ""
    params = None
    if limit is not None:
        limit_clause = " LIMIT %s"
        params = (limit,)

    query = base_query + order_clause + limit_clause

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


def fetch_training_data_from_predictions(total_limit: int = 20000) -> pd.DataFrame:
    """
    Fetch training data from predictions table.
    
    Priority: Fraud predictions first, then non-fraud predictions.
    Uses actual_label if available, otherwise falls back to prediction column.
    
    Args:
        total_limit: Maximum total rows to fetch (default 20,000)
    
    Returns:
        DataFrame with transaction features and isFraud column for training
    """
    # Query to get fraud predictions first (prediction = TRUE)
    fraud_query = """
        SELECT 
            p.prediction,
            p.actual_label,
            p.predict_proba,
            t.step, t.type, t.amount, 
            t."nameOrig", t."oldbalanceOrg", t."newbalanceOrig",
            t."nameDest", t."oldbalanceDest", t."newbalanceDest",
            t."isFlaggedFraud"
        FROM predictions p
        JOIN all_transactions t ON p.transaction_id = t.id
        WHERE p.prediction = TRUE
        ORDER BY p.prediction_time DESC
        LIMIT %s
    """
    
    # Query to get non-fraud predictions (prediction = FALSE)
    non_fraud_query = """
        SELECT 
            p.prediction,
            p.actual_label,
            p.predict_proba,
            t.step, t.type, t.amount, 
            t."nameOrig", t."oldbalanceOrg", t."newbalanceOrig",
            t."nameDest", t."oldbalanceDest", t."newbalanceDest",
            t."isFlaggedFraud"
        FROM predictions p
        JOIN all_transactions t ON p.transaction_id = t.id
        WHERE p.prediction = FALSE
        ORDER BY p.prediction_time DESC
        LIMIT %s
    """
    
    with get_connection() as conn:
        # Fetch fraud predictions first
        fraud_df = pd.read_sql_query(fraud_query, conn, params=(total_limit,))
        
        # Calculate remaining limit for non-fraud
        remaining_limit = total_limit - len(fraud_df)
        
        if remaining_limit > 0:
            non_fraud_df = pd.read_sql_query(non_fraud_query, conn, params=(remaining_limit,))
        else:
            non_fraud_df = pd.DataFrame()
    
    # Combine fraud and non-fraud
    if not non_fraud_df.empty:
        combined_df = pd.concat([fraud_df, non_fraud_df], ignore_index=True)
    else:
        combined_df = fraud_df
    
    if combined_df.empty:
        return pd.DataFrame()
    
    # Create isFraud column based on actual_label (if available) or prediction
    # If actual_label is not None, use it; otherwise use prediction
    combined_df['isFraud'] = combined_df.apply(
        lambda row: row['actual_label'] if row['actual_label'] is not None else row['prediction'],
        axis=1
    )
    
    # Convert to int for training
    combined_df['isFraud'] = combined_df['isFraud'].astype(int)
    
    # Drop the prediction and actual_label columns as they're not needed for training
    training_df = combined_df.drop(columns=['prediction', 'actual_label', 'predict_proba'])
    
    return training_df

