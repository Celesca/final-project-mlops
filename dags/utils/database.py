"""
Database utilities for storing fraud detection prediction results.

This module handles:
1. Master transaction table (all ingested transactions)
2. Correct predictions table (TP and TN cases)
3. Incorrect predictions table (FP and FN cases with predict_proba)
"""
from decimal import Decimal
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime, date
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
                transaction_id INTEGER NOT NULL REFERENCES all_transactions(id) ON DELETE CASCADE,
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
                transaction_id INTEGER NOT NULL REFERENCES all_transactions(id) ON DELETE CASCADE,
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
            CREATE INDEX IF NOT EXISTS idx_correct_predictions_transaction 
            ON correct_predictions(transaction_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_incorrect_predictions_time 
            ON incorrect_predictions(prediction_time)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_incorrect_predictions_transaction 
            ON incorrect_predictions(transaction_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_incorrect_predictions_type 
            ON incorrect_predictions(prediction, actual_label)
        """)

    print("✅ Fraud prediction tables ensured in Postgres")


def save_prediction(
    transaction_id: int,
    prediction: bool,
    actual_label: bool,
    predict_proba: float,
    prediction_time: Optional[str] = None,
) -> None:
    """
    Save a prediction result to the predictions table.
    
    The actual_label is initially stored as False (masked/unknown) and can be
    updated later via the /PUT endpoint when the true label is confirmed.
    
    Args:
        transaction_id: Foreign key to the stored transaction
        prediction: Model prediction (True/False for is_fraud)
        actual_label: Ignored - always stored as False initially
        predict_proba: Probability score from model
        prediction_time: ISO timestamp (defaults to current time)
    """
    if transaction_id is None:
        raise ValueError("transaction_id is required to save a prediction record.")

    if prediction_time is None:
        prediction_time = datetime.utcnow().isoformat()
    
    pred_int = 1 if prediction else 0
    # actual_label is always 0 (False/unknown) initially - will be updated manually later
    actual_int = 0
    
    with get_cursor(commit=True) as cursor:
        # Store in incorrect_predictions table (which has predict_proba column)
        # This table now serves as the main predictions table
        cursor.execute(
            """
            INSERT INTO incorrect_predictions 
            (transaction_id, prediction, actual_label, predict_proba, prediction_time)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (transaction_id, pred_int, actual_int, float(predict_proba), prediction_time),
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

TRANSACTION_KEY_COLUMNS = [
    "ingest_date",
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "oldbalanceOrg",
    "oldbalanceDest",
    "newbalanceOrig",
    "newbalanceDest",
]

TRANSACTION_RESULT_COLUMNS = MASTER_TX_COLUMNS + ["ingest_date", "source_file"]


def _alias_column(column: str, table_alias: str = "at") -> str:
    # If the logical column name contains uppercase letters we must quote
    # the identifier in SQL so Postgres matches the exact mixed-case name
    # (some deployments use quoted mixed-case column names). For plain
    # lowercase names use the unquoted form.
    if column.islower():
        return f'{table_alias}.{column} AS "{column}"'
    else:
        return f'{table_alias}."{column}" AS "{column}"'


TRANSACTION_RESULT_SELECT = ", ".join(_alias_column(col) for col in TRANSACTION_RESULT_COLUMNS)
TRANSACTION_LOOKUP_SELECT = ", ".join(_alias_column(col) for col in TRANSACTION_KEY_COLUMNS)


def _normalize_key_value(value: Any):
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _normalize_key_value(value.item())
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return float(value)
    return value


def build_transaction_key(transaction: Dict[str, Any], ingest_date: Optional[str] = None) -> Tuple[Any, ...]:
    """
    Build a deterministic key for mapping transactions back to their DB row.
    """
    key_parts: List[Any] = []
    for column in TRANSACTION_KEY_COLUMNS:
        if column == "ingest_date":
            key_parts.append(_normalize_key_value(transaction.get(column, ingest_date)))
            continue
        key_parts.append(_normalize_key_value(transaction.get(column)))
    return tuple(key_parts)


def get_transaction_lookup_for_ingest(ingest_date: str) -> Dict[Tuple[Any, ...], int]:
    """
    Build a lookup table of transaction keys -> database IDs for a given ingest date.
    """
    query = f"""
        SELECT at.id, {TRANSACTION_LOOKUP_SELECT}
        FROM all_transactions at
        WHERE at.ingest_date = %s
    """
    with get_cursor(dict_cursor=True) as cursor:
        cursor.execute(query, (ingest_date,))
        rows = cursor.fetchall()

    lookup: Dict[Tuple[Any, ...], int] = {}
    for row in rows:
        lookup[build_transaction_key(row)] = row["id"]
    return lookup


def _extract_master_row(transaction: Dict) -> List:
    """Extract ordered column values for master table."""
    row = []
    for col in MASTER_TX_COLUMNS:
        value = transaction.get(col)
        if col in {"isFraud", "isFlaggedFraud"} and value is not None:
            value = bool(value)
        row.append(value)
    return row


def save_transaction_record(
    transaction: Dict,
    ingest_date: str,
    source_file: Optional[str] = None,
) -> int:
    """
    Persist a raw transaction into the all_transactions table and return its ID.
    """
    row = _extract_master_row(transaction)
    row.extend([ingest_date, source_file])

    with get_cursor(commit=True) as cursor:
        cursor.execute(
            """
            INSERT INTO all_transactions (
                step, type, amount, "nameOrig", "oldbalanceOrg", "newbalanceOrig",
                "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
                ingest_date, source_file
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            row,
        )
        new_id = cursor.fetchone()[0]
    return new_id


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
                step, type, amount, "nameOrig", "oldbalanceOrg", "newbalanceOrig",
                "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
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
    Retrieve predictions where prediction = 0 (non-fraud predictions).
    
    Args:
        limit: Optional limit on number of records
    
    Returns:
        List of dictionaries with prediction records
    """
    query = f"""
        SELECT 
            ip.id,
            ip.transaction_id,
            ip.prediction,
            ip.actual_label,
            ip.predict_proba,
            ip.prediction_time,
            ip.created_at,
            {TRANSACTION_RESULT_SELECT}
        FROM incorrect_predictions ip
        JOIN all_transactions at ON at.id = ip.transaction_id
        WHERE ip.prediction = 0
        ORDER BY ip.prediction_time DESC
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
            "transaction_id": row["transaction_id"],
            "transaction_data": {col: row.get(col) for col in TRANSACTION_RESULT_COLUMNS},
            "prediction": bool(row["prediction"]),
            "actual_label": bool(row["actual_label"]),
            "predict_proba": row["predict_proba"],
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
    # Use quoted identifiers for mixed-case columns to match the DB init SQL
    query = """
        SELECT id, step, type, amount, "nameOrig", "oldbalanceOrg", "newbalanceOrig",
               "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
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
            "nameOrig": row.get("nameOrig") if "nameOrig" in row else row.get("nameorig"),
            "oldbalanceOrg": row.get("oldbalanceOrg") if "oldbalanceOrg" in row else row.get("oldbalanceorg"),
            "newbalanceOrig": row.get("newbalanceOrig") if "newbalanceOrig" in row else row.get("newbalanceorig"),
            "nameDest": row.get("nameDest") if "nameDest" in row else row.get("namedest"),
            "oldbalanceDest": row.get("oldbalanceDest") if "oldbalanceDest" in row else row.get("oldbalancedest"),
            "newbalanceDest": row.get("newbalanceDest") if "newbalanceDest" in row else row.get("newbalancedest"),
            "isFraud": bool(row.get("isFraud") if "isFraud" in row else row.get("isfraud")),
            "isFlaggedFraud": bool(row.get("isFlaggedFraud") if "isFlaggedFraud" in row else row.get("isflaggedfraud")),
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
    Retrieve predictions where prediction = 1 (fraud predictions).
    
    Args:
        limit: Optional limit on number of records
    
    Returns:
        List of dictionaries with prediction records including predict_proba
    """
    query = f"""
        SELECT 
            ip.id,
            ip.transaction_id,
            ip.prediction,
            ip.actual_label,
            ip.predict_proba,
            ip.prediction_time,
            ip.created_at,
            {TRANSACTION_RESULT_SELECT}
        FROM incorrect_predictions ip
        JOIN all_transactions at ON at.id = ip.transaction_id
        WHERE ip.prediction = 1
        ORDER BY ip.prediction_time DESC
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
            "transaction_id": row["transaction_id"],
            "transaction_data": {col: row.get(col) for col in TRANSACTION_RESULT_COLUMNS},
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
        Dictionary with counts of fraud predictions, non-fraud predictions, 
        and confirmed labels
    """
    with get_cursor() as cursor:
        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM incorrect_predictions")
        total_count = cursor.fetchone()[0]
        
        # Fraud predictions (prediction = 1)
        cursor.execute("SELECT COUNT(*) FROM incorrect_predictions WHERE prediction = 1")
        fraud_predictions = cursor.fetchone()[0]
        
        # Non-fraud predictions (prediction = 0)
        cursor.execute("SELECT COUNT(*) FROM incorrect_predictions WHERE prediction = 0")
        non_fraud_predictions = cursor.fetchone()[0]
        
        # Confirmed fraud (actual_label = 1)
        cursor.execute("SELECT COUNT(*) FROM incorrect_predictions WHERE actual_label = 1")
        confirmed_fraud = cursor.fetchone()[0]
        
        # Unconfirmed (actual_label = 0, which is the initial masked state)
        cursor.execute("SELECT COUNT(*) FROM incorrect_predictions WHERE actual_label = 0")
        unconfirmed = cursor.fetchone()[0]
    
    return {
        "total_predictions": total_count,
        "fraud_predictions": fraud_predictions,
        "non_fraud_predictions": non_fraud_predictions,
        "confirmed_fraud": confirmed_fraud,
        "unconfirmed": unconfirmed,
    }

