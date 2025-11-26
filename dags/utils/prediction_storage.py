"""
Helper functions for storing model predictions in the database.

This module provides a convenient interface for saving predictions
after model inference. In real-world scenarios, actual_label is unknown
at prediction time and will be NULL initially.
"""
from typing import Dict, Optional
from datetime import datetime
from dags.utils.database import save_prediction, init_prediction_db, get_prediction_stats


def store_prediction_result(
    transaction_id: int,
    prediction: bool,
    predict_proba: float,
    prediction_time: Optional[datetime] = None
) -> int:
    """
    Store a prediction result in the predictions table.
    
    In real-world ML workflows, the actual_label is unknown at prediction time.
    It will be set to NULL initially and updated later via human review.
    
    Args:
        transaction_id: ID of the transaction stored in all_transactions
        prediction: Model prediction (True/False for is_fraud)
        predict_proba: Probability score from model (0.0 to 1.0)
        prediction_time: Optional timestamp (defaults to current UTC time)
    
    Returns:
        ID of the saved prediction record
    
    Example:
        >>> prediction_id = store_prediction_result(
        ...     transaction_id=123,
        ...     prediction=True,
        ...     predict_proba=0.85
        ... )
    """
    # Ensure database is initialized
    try:
        init_prediction_db()
    except Exception:
        pass  # Database might already exist
    
    if prediction_time is None:
        prediction_time = datetime.utcnow()
    
    return save_prediction(
        transaction_id=transaction_id,
        prediction=prediction,
        predict_proba=predict_proba,
        prediction_time=prediction_time
    )


def get_model_performance_stats() -> Dict:
    """
    Get performance statistics from stored predictions.
    
    Returns:
        Dictionary with:
        - total_predictions: Total number of predictions
        - predicted_frauds: Number predicted as fraud
        - predicted_non_frauds: Number predicted as non-fraud
        - labeled_predictions: Predictions with actual_label set
        - unlabeled_predictions: Predictions awaiting human review
        - accuracy: Accuracy for labeled predictions (or None)
        - total_transactions: Total transactions in database
    """
    return get_prediction_stats()
