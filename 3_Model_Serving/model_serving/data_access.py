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

