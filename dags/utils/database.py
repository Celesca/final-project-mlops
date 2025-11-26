"""
Database utilities for Fraud Detection MLOps Pipeline.
Handles PostgreSQL operations for storing transactions and predictions.
Uses a single 'predictions' table where actual_label is NULL initially.
"""

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('FRAUD_DB_HOST', 'fraud-db'),
    'port': os.getenv('FRAUD_DB_PORT', '5432'),
    'database': os.getenv('FRAUD_DB_NAME', 'fraud_detection'),
    'user': os.getenv('FRAUD_DB_USER', 'fraud_user'),
    'password': os.getenv('FRAUD_DB_PASSWORD', 'fraud_password')
}

# Transaction columns as they appear in the database (with quotes for mixed case)
TRANSACTION_COLUMNS = [
    'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
    'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud'
]


def get_connection():
    """Create and return a database connection."""
    return psycopg2.connect(**DB_CONFIG)


def init_prediction_db():
    """
    Initialize the prediction database with required tables.
    Creates:
    - all_transactions: stores transaction records
    - predictions: stores predictions with actual_label initially NULL
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Create all_transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS all_transactions (
                id SERIAL PRIMARY KEY,
                step INTEGER NOT NULL,
                type VARCHAR(50) NOT NULL,
                amount DOUBLE PRECISION NOT NULL,
                "nameOrig" VARCHAR(100) NOT NULL,
                "oldbalanceOrg" DOUBLE PRECISION NOT NULL,
                "newbalanceOrig" DOUBLE PRECISION NOT NULL,
                "nameDest" VARCHAR(100) NOT NULL,
                "oldbalanceDest" DOUBLE PRECISION NOT NULL,
                "newbalanceDest" DOUBLE PRECISION NOT NULL,
                "isFraud" BOOLEAN NOT NULL,
                "isFlaggedFraud" BOOLEAN NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create single predictions table with actual_label as nullable
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                transaction_id INTEGER NOT NULL REFERENCES all_transactions(id) ON DELETE CASCADE,
                prediction BOOLEAN NOT NULL,
                actual_label BOOLEAN,
                predict_proba DOUBLE PRECISION NOT NULL,
                prediction_time TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id 
            ON predictions(transaction_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_prediction 
            ON predictions(prediction)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_actual_label 
            ON predictions(actual_label)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_prediction_time 
            ON predictions(prediction_time)
        ''')
        
        conn.commit()
        logger.info("Prediction database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing prediction database: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def save_prediction(
    transaction_id: int,
    prediction: bool,
    predict_proba: float,
    prediction_time: datetime
) -> int:
    """
    Save a prediction to the predictions table.
    actual_label is always NULL initially - it will be updated later via human review.
    
    Args:
        transaction_id: ID of the transaction
        prediction: Model's prediction (True for fraud, False for non-fraud)
        predict_proba: Probability score from model
        prediction_time: When prediction was made
    
    Returns:
        ID of the saved prediction record
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (
                transaction_id, prediction, actual_label, predict_proba, prediction_time
            )
            VALUES (%s, %s, NULL, %s, %s)
            RETURNING id
        ''', (transaction_id, prediction, predict_proba, prediction_time))
        
        prediction_id = cursor.fetchone()[0]
        conn.commit()
        
        logger.info(f"Saved prediction {prediction_id} for transaction {transaction_id}")
        return prediction_id
        
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def save_transactions_bulk(transactions: List[Dict[str, Any]]) -> List[int]:
    """
    Save multiple transactions in bulk for better performance.
    
    Args:
        transactions: List of transaction dictionaries
        
    Returns:
        List of saved transaction IDs
    """
    if not transactions:
        return []
    
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        columns = ', '.join([f'"{col}"' if any(c.isupper() for c in col) else col 
                            for col in TRANSACTION_COLUMNS])
        
        # Prepare values for bulk insert
        values_list = []
        for tx in transactions:
            row = []
            for col in TRANSACTION_COLUMNS:
                val = tx.get(col)
                if col in ('isFraud', 'isFlaggedFraud'):
                    val = bool(val) if val is not None else False
                row.append(val)
            values_list.append(tuple(row))
        
        # Use execute_values for efficient bulk insert
        query = f'''
            INSERT INTO all_transactions ({columns})
            VALUES %s
            RETURNING id
        '''
        
        cursor.execute(f'''
            INSERT INTO all_transactions ({columns})
            VALUES {','.join(['(' + ','.join(['%s'] * len(TRANSACTION_COLUMNS)) + ')'] * len(values_list))}
            RETURNING id
        ''', [val for row in values_list for val in row])
        
        ids = [row[0] for row in cursor.fetchall()]
        conn.commit()
        
        logger.info(f"Saved {len(ids)} transactions in bulk")
        return ids
        
    except Exception as e:
        logger.error(f"Error saving transactions in bulk: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def get_transactions(
    limit: Optional[int] = None,
    offset: int = 0,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve transactions from the database.
    
    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip
        start_date: Filter by created_at >= start_date
        end_date: Filter by created_at <= end_date
        
    Returns:
        List of transaction dictionaries
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = 'SELECT * FROM all_transactions WHERE 1=1'
        params = []
        
        if start_date:
            query += ' AND created_at >= %s'
            params.append(start_date)
        
        if end_date:
            query += ' AND created_at <= %s'
            params.append(end_date)
        
        query += ' ORDER BY created_at DESC'
        
        if limit:
            query += ' LIMIT %s'
            params.append(limit)
        
        if offset:
            query += ' OFFSET %s'
            params.append(offset)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        raise
    finally:
        if conn:
            conn.close()


def get_all_predictions(
    limit: Optional[int] = None,
    offset: int = 0,
    only_unlabeled: bool = False
) -> List[Dict[str, Any]]:
    """
    Get all predictions with their associated transactions.
    
    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip
        only_unlabeled: If True, return only predictions where actual_label is NULL
        
    Returns:
        List of prediction records with transaction data
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = '''
            SELECT 
                p.id as prediction_id,
                p.transaction_id,
                p.prediction,
                p.actual_label,
                p.predict_proba,
                p.prediction_time,
                p.created_at as prediction_created_at,
                t.*
            FROM predictions p
            JOIN all_transactions t ON p.transaction_id = t.id
            WHERE 1=1
        '''
        params = []
        
        if only_unlabeled:
            query += ' AND p.actual_label IS NULL'
        
        query += ' ORDER BY p.prediction_time DESC'
        
        if limit:
            query += ' LIMIT %s'
            params.append(limit)
        
        if offset:
            query += ' OFFSET %s'
            params.append(offset)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise
    finally:
        if conn:
            conn.close()


def get_frauds(
    limit: Optional[int] = None,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Get predictions where prediction = True (predicted as fraud).
    
    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip
        
    Returns:
        List of predictions predicted as fraud
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = '''
            SELECT 
                p.id as prediction_id,
                p.transaction_id,
                p.prediction,
                p.actual_label,
                p.predict_proba,
                p.prediction_time,
                p.created_at as prediction_created_at,
                t.*
            FROM predictions p
            JOIN all_transactions t ON p.transaction_id = t.id
            WHERE p.prediction = TRUE
            ORDER BY p.prediction_time DESC
        '''
        params = []
        
        if limit:
            query += ' LIMIT %s'
            params.append(limit)
        
        if offset:
            query += ' OFFSET %s'
            params.append(offset)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error(f"Error getting frauds: {e}")
        raise
    finally:
        if conn:
            conn.close()


def get_non_frauds(
    limit: Optional[int] = None,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Get predictions where prediction = False (predicted as non-fraud).
    
    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip
        
    Returns:
        List of predictions predicted as non-fraud
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = '''
            SELECT 
                p.id as prediction_id,
                p.transaction_id,
                p.prediction,
                p.actual_label,
                p.predict_proba,
                p.prediction_time,
                p.created_at as prediction_created_at,
                t.*
            FROM predictions p
            JOIN all_transactions t ON p.transaction_id = t.id
            WHERE p.prediction = FALSE
            ORDER BY p.prediction_time DESC
        '''
        params = []
        
        if limit:
            query += ' LIMIT %s'
            params.append(limit)
        
        if offset:
            query += ' OFFSET %s'
            params.append(offset)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error(f"Error getting non-frauds: {e}")
        raise
    finally:
        if conn:
            conn.close()


def update_actual_label(prediction_id: int, actual_label: bool) -> bool:
    """
    Update the actual_label for a prediction after human review.
    
    Args:
        prediction_id: ID of the prediction to update
        actual_label: The actual label (True for fraud, False for non-fraud)
        
    Returns:
        True if update was successful, False otherwise
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE predictions 
            SET actual_label = %s
            WHERE id = %s
        ''', (actual_label, prediction_id))
        
        rows_affected = cursor.rowcount
        conn.commit()
        
        if rows_affected > 0:
            logger.info(f"Updated actual_label for prediction {prediction_id} to {actual_label}")
            return True
        else:
            logger.warning(f"No prediction found with id {prediction_id}")
            return False
        
    except Exception as e:
        logger.error(f"Error updating actual_label: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def get_prediction_stats() -> Dict[str, Any]:
    """
    Get statistics about predictions.
    
    Returns:
        Dictionary containing prediction statistics
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get total counts
        cursor.execute('''
            SELECT 
                COUNT(*) as total_predictions,
                COUNT(CASE WHEN prediction = TRUE THEN 1 END) as predicted_frauds,
                COUNT(CASE WHEN prediction = FALSE THEN 1 END) as predicted_non_frauds,
                COUNT(CASE WHEN actual_label IS NOT NULL THEN 1 END) as labeled_predictions,
                COUNT(CASE WHEN actual_label IS NULL THEN 1 END) as unlabeled_predictions
            FROM predictions
        ''')
        counts = dict(cursor.fetchone())
        
        # Get accuracy for labeled predictions
        cursor.execute('''
            SELECT 
                COUNT(CASE WHEN prediction = actual_label THEN 1 END) as correct,
                COUNT(*) as total
            FROM predictions
            WHERE actual_label IS NOT NULL
        ''')
        accuracy_data = cursor.fetchone()
        
        if accuracy_data['total'] > 0:
            accuracy = accuracy_data['correct'] / accuracy_data['total']
        else:
            accuracy = None
        
        # Get transaction count
        cursor.execute('SELECT COUNT(*) as total FROM all_transactions')
        transaction_count = cursor.fetchone()['total']
        
        return {
            **counts,
            'accuracy': accuracy,
            'total_transactions': transaction_count
        }
        
    except Exception as e:
        logger.error(f"Error getting prediction stats: {e}")
        raise
    finally:
        if conn:
            conn.close()


def get_labeled_predictions_for_training() -> List[Dict[str, Any]]:
    """
    Get all predictions that have been labeled (actual_label is not NULL).
    These can be used for retraining the model.
    
    Returns:
        List of labeled predictions with transaction data
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute('''
            SELECT 
                p.id as prediction_id,
                p.transaction_id,
                p.prediction,
                p.actual_label,
                p.predict_proba,
                p.prediction_time,
                t.*
            FROM predictions p
            JOIN all_transactions t ON p.transaction_id = t.id
            WHERE p.actual_label IS NOT NULL
            ORDER BY p.prediction_time DESC
        ''')
        
        results = cursor.fetchall()
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error(f"Error getting labeled predictions: {e}")
        raise
    finally:
        if conn:
            conn.close()


def get_misclassified_predictions() -> List[Dict[str, Any]]:
    """
    Get predictions where the model's prediction differs from the actual label.
    
    Returns:
        List of misclassified predictions
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute('''
            SELECT 
                p.id as prediction_id,
                p.transaction_id,
                p.prediction,
                p.actual_label,
                p.predict_proba,
                p.prediction_time,
                t.*
            FROM predictions p
            JOIN all_transactions t ON p.transaction_id = t.id
            WHERE p.actual_label IS NOT NULL 
              AND p.prediction != p.actual_label
            ORDER BY p.prediction_time DESC
        ''')
        
        results = cursor.fetchall()
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error(f"Error getting misclassified predictions: {e}")
        raise
    finally:
        if conn:
            conn.close()


def drop_all_tables():
    """Drop all tables (for testing/reset purposes)."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DROP TABLE IF EXISTS predictions CASCADE')
        cursor.execute('DROP TABLE IF EXISTS all_transactions CASCADE')
        
        conn.commit()
        logger.info("All tables dropped successfully")
        
    except Exception as e:
        logger.error(f"Error dropping tables: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def build_transaction_key(transaction: Dict[str, Any], ingest_date: str = None) -> str:
    """
    Build a unique key for a transaction to use in lookups.
    
    Args:
        transaction: Transaction dictionary
        ingest_date: Optional ingest date to include in key
        
    Returns:
        String key that uniquely identifies the transaction
    """
    step = transaction.get('step', '')
    tx_type = transaction.get('type', '')
    amount = transaction.get('amount', '')
    name_orig = transaction.get('nameOrig', '')
    name_dest = transaction.get('nameDest', '')
    
    key_parts = [str(step), tx_type, str(amount), name_orig, name_dest]
    if ingest_date:
        key_parts.append(ingest_date)
    
    return '|'.join(key_parts)


def get_transaction_lookup_for_ingest(ingest_date: str) -> Dict[str, int]:
    """
    Get a lookup dictionary of transaction keys to IDs for a given ingest date.
    
    Args:
        ingest_date: The date to filter transactions by (YYYY-MM-DD format)
        
    Returns:
        Dictionary mapping transaction keys to their IDs
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute('''
            SELECT id, step, type, amount, "nameOrig", "nameDest"
            FROM all_transactions
            WHERE DATE(created_at) = %s
        ''', (ingest_date,))
        
        results = cursor.fetchall()
        lookup = {}
        
        for row in results:
            key = build_transaction_key(dict(row), ingest_date)
            lookup[key] = row['id']
        
        return lookup
        
    except Exception as e:
        logger.error(f"Error getting transaction lookup: {e}")
        raise
    finally:
        if conn:
            conn.close()


def save_transaction_record(
    transaction: Dict[str, Any],
    ingest_date: str = None,
    source_file: str = None
) -> int:
    """
    Save a single transaction record to the database.
    
    Args:
        transaction: Dictionary containing transaction fields
        ingest_date: Optional ingest date
        source_file: Optional source file name
        
    Returns:
        ID of the saved transaction
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        columns = ', '.join([f'"{col}"' if any(c.isupper() for c in col) else col 
                            for col in TRANSACTION_COLUMNS])
        placeholders = ', '.join(['%s'] * len(TRANSACTION_COLUMNS))
        
        values = []
        for col in TRANSACTION_COLUMNS:
            val = transaction.get(col)
            # Convert boolean fields
            if col in ('isFraud', 'isFlaggedFraud'):
                val = bool(val) if val is not None else False
            values.append(val)
        
        cursor.execute(f'''
            INSERT INTO all_transactions ({columns})
            VALUES ({placeholders})
            RETURNING id
        ''', values)
        
        transaction_id = cursor.fetchone()[0]
        conn.commit()
        
        logger.info(f"Saved transaction {transaction_id}")
        return transaction_id
        
    except Exception as e:
        logger.error(f"Error saving transaction: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


