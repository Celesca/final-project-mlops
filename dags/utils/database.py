"""
Database utilities for storing fraud detection prediction results.

This module handles:
1. Master transaction table (all ingested transactions)
2. Correct predictions table (TP and TN cases)
3. Incorrect predictions table (FP and FN cases with predict_proba)
"""
import json
from typing import Dict, Optional, List, Any
from datetime import datetime
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from dags.config import FRAUD_DB_CONFIG


def _get_connection():
    """Create a new psycopg2 connection using config/env variables."""
    return psycopg2.connect(
        host=FRAUD_DB_CONFIG["host"],
        port=FRAUD_DB_CONFIG["port"],
        dbname=FRAUD_DB_CONFIG["dbname"],
        user=FRAUD_DB_CONFIG["user"],
        password=FRAUD_DB_CONFIG["password"],
    )


@contextmanager
def get_cursor(commit: bool = False, dict_cursor: bool = False):
    conn = _get_connection()
    cursor_factory = RealDictCursor if dict_cursor else None
    cur = conn.cursor(cursor_factory=cursor_factory)
    try:
        yield cur
        if commit:
            conn.commit()
    finally:
        cur.close()
        conn.close()


def init_prediction_db(_: Optional[Any] = None) -> None:
    """
    Initialize the prediction database with two tables:
    1. correct_predictions - for TP and TN cases
    2. incorrect_predictions - for FP and FN cases (includes predict_proba)
    """
    with get_cursor(commit=True) as cursor:
        # Table 1: All transactions (raw ingestion storage with explicit columns)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS all_transactions (
                id SERIAL PRIMARY KEY,
                step INTEGER,
                type TEXT,
                amount DOUBLE PRECISION,
                nameOrig TEXT,
                oldbalanceOrg DOUBLE PRECISION,
                newbalanceOrig DOUBLE PRECISION,
                nameDest TEXT,
                oldbalanceDest DOUBLE PRECISION,
                newbalanceDest DOUBLE PRECISION,
                isFraud INTEGER,
                isFlaggedFraud INTEGER,
                ingest_date DATE NOT NULL,
                source_file TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table 2: Correct predictions (TP and TN)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS correct_predictions (
                id SERIAL PRIMARY KEY,
                transaction_data JSONB NOT NULL,
                prediction INTEGER NOT NULL,
                actual_label INTEGER NOT NULL,
                prediction_time TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table 3: Incorrect predictions (FP and FN) with predict_proba
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS incorrect_predictions (
                id SERIAL PRIMARY KEY,
                transaction_data JSONB NOT NULL,
                prediction INTEGER NOT NULL,
                actual_label INTEGER NOT NULL,
                predict_proba DOUBLE PRECISION NOT NULL,
                prediction_time TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_all_transactions_ingest 
            ON all_transactions(ingest_date)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_correct_predictions_time 
            ON correct_predictions(prediction_time)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_incorrect_predictions_time 
            ON incorrect_predictions(prediction_time)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_incorrect_predictions_type 
            ON incorrect_predictions(prediction, actual_label)
        """)

    print("✅ Fraud prediction tables ensured in Postgres")


def save_prediction(
    transaction: Dict,
    prediction: bool,
    actual_label: bool,
    predict_proba: float,
    prediction_time: Optional[str] = None,
) -> None:
    """
    Save a prediction result to the appropriate table based on correctness.
    
    Args:
        transaction: Transaction data dictionary
        prediction: Model prediction (True/False for is_fraud)
        actual_label: Actual label (True/False for is_fraud)
        predict_proba: Probability score from model
        prediction_time: ISO timestamp (defaults to current time)
    """
    if prediction_time is None:
        prediction_time = datetime.utcnow().isoformat()
    
    pred_int = 1 if prediction else 0
    actual_int = 1 if actual_label else 0
    transaction_json = json.dumps(transaction)
    is_correct = (prediction == actual_label)
    
    with get_cursor(commit=True) as cursor:
        if is_correct:
            cursor.execute(
                """
                INSERT INTO correct_predictions 
                (transaction_data, prediction, actual_label, prediction_time)
                VALUES (%s, %s, %s, %s)
                """,
                (transaction_json, pred_int, actual_int, prediction_time),
            )
        else:
            cursor.execute(
                """
                INSERT INTO incorrect_predictions 
                (transaction_data, prediction, actual_label, predict_proba, prediction_time)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (transaction_json, pred_int, actual_int, float(predict_proba), prediction_time),
            )


MASTER_TX_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def _extract_master_row(transaction: Dict) -> List:
    """Extract ordered column values for master table."""
    row = []
    for col in MASTER_TX_COLUMNS:
        value = transaction.get(col)
        if col in {"isFraud", "isFlaggedFraud"} and value is not None:
            value = int(bool(value))
        row.append(value)
    return row


def save_transaction_record(
    transaction: Dict,
    ingest_date: str,
    source_file: Optional[str] = None,
) -> None:
    """
    Persist a raw transaction into the all_transactions table.
    """
    row = _extract_master_row(transaction)
    row.extend([ingest_date, source_file])

    with get_cursor(commit=True) as cursor:
        cursor.execute(
            """
            INSERT INTO all_transactions (
                step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
                nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud,
                ingest_date, source_file
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            row,
        )


def save_transactions_bulk(
    transactions: List[Dict],
    ingest_date: str,
    source_file: Optional[str] = None,
) -> None:
    """
    Persist multiple transactions efficiently.
    """
    if not transactions:
        return

    payload = []
    for txn in transactions:
        row = _extract_master_row(txn)
        row.extend([ingest_date, source_file])
        payload.append(tuple(row))

    with get_cursor(commit=True) as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO all_transactions (
                step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
                nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud,
                ingest_date, source_file
            )
            VALUES %s
            """,
            payload,
        )


def get_correct_predictions(
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    Retrieve correct predictions from the database.
    
    Args:
        limit: Optional limit on number of records
    
    Returns:
        List of dictionaries with prediction records
    """
    query = """
        SELECT id, transaction_data, prediction, actual_label, prediction_time, created_at
        FROM correct_predictions
        ORDER BY prediction_time DESC
    """
    params: List[Any] = []
    if limit:
        query += " LIMIT %s"
        params.append(limit)
    
    with get_cursor(dict_cursor=True) as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()
    
    result = []
    for row in rows:
        result.append({
            "id": row["id"],
            "transaction_data": row["transaction_data"] or {},
            "prediction": bool(row["prediction"]),
            "actual_label": bool(row["actual_label"]),
            "prediction_time": row["prediction_time"],
            "created_at": row["created_at"],
        })
    
    return result


def get_transactions(
    ingest_date: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    Retrieve transactions from the all_transactions table.

    Args:
        ingest_date: Optional YYYY-MM-DD filter
        limit: Optional limit on number of records

    Returns:
        List of transaction dictionaries
    """
    query = """
        SELECT id, step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
               nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud,
               ingest_date, source_file, created_at
        FROM all_transactions
    """
    params: List[Any] = []
    clauses = []

    if ingest_date:
        clauses.append("ingest_date = %s")
        params.append(ingest_date)

    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    query += " ORDER BY created_at DESC"

    if limit:
        query += " LIMIT %s"
        params.append(limit)

    with get_cursor(dict_cursor=True) as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()

    return [
        {
            "id": row["id"],
            "step": row["step"],
            "type": row["type"],
            "amount": row["amount"],
            "nameOrig": row["nameorig"],
            "oldbalanceOrg": row["oldbalanceorg"],
            "newbalanceOrig": row["newbalanceorig"],
            "nameDest": row["namedest"],
            "oldbalanceDest": row["oldbalancedest"],
            "newbalanceDest": row["newbalancedest"],
            "isFraud": row["isfraud"],
            "isFlaggedFraud": row["isflaggedfraud"],
            "ingest_date": row["ingest_date"],
            "source_file": row["source_file"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_incorrect_predictions(
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    Retrieve incorrect predictions (FP and FN) from the database.
    
    Args:
        limit: Optional limit on number of records
    
    Returns:
        List of dictionaries with prediction records including predict_proba
    """
    query = """
        SELECT id, transaction_data, prediction, actual_label, predict_proba, prediction_time, created_at
        FROM incorrect_predictions
        ORDER BY prediction_time DESC
    """
    params: List[Any] = []
    if limit:
        query += " LIMIT %s"
        params.append(limit)
    
    with get_cursor(dict_cursor=True) as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()
    
    result = []
    for row in rows:
        result.append({
            "id": row["id"],
            "transaction_data": row["transaction_data"] or {},
            "prediction": bool(row["prediction"]),
            "actual_label": bool(row["actual_label"]),
            "predict_proba": row["predict_proba"],
            "prediction_time": row["prediction_time"],
            "created_at": row["created_at"],
        })
    
    return result


def get_prediction_stats() -> Dict:
    """
    Get statistics about predictions stored in the database.
    
    Returns:
        Dictionary with counts of TP, TN, FP, FN, and totals
    """
    with get_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM correct_predictions")
        correct_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM incorrect_predictions")
        incorrect_count = cursor.fetchone()[0]
        
        cursor.execute(
            """
            SELECT COUNT(*) FROM incorrect_predictions 
            WHERE prediction = 1 AND actual_label = 0
            """
        )
        fp_count = cursor.fetchone()[0]
        
        cursor.execute(
            """
            SELECT COUNT(*) FROM incorrect_predictions 
            WHERE prediction = 0 AND actual_label = 1
            """
        )
        fn_count = cursor.fetchone()[0]
        
        cursor.execute(
            """
            SELECT COUNT(*) FROM correct_predictions 
            WHERE prediction = 1 AND actual_label = 1
            """
        )
        tp_count = cursor.fetchone()[0]
        
        cursor.execute(
            """
            SELECT COUNT(*) FROM correct_predictions 
            WHERE prediction = 0 AND actual_label = 0
            """
        )
        tn_count = cursor.fetchone()[0]
    
    total = correct_count + incorrect_count
    accuracy = (correct_count / total * 100) if total > 0 else 0.0
    
    return {
        "total_predictions": total,
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "true_positives": tp_count,
        "true_negatives": tn_count,
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "accuracy": round(accuracy, 2)
    }

