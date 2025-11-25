"""
Helper functions for storing model predictions in the database.

This module provides a convenient interface for saving predictions
after model inference, automatically routing to the correct table.
"""
from typing import Dict, Optional
from datetime import datetime
from dags.utils.database import save_prediction, init_prediction_db, get_prediction_stats


def store_prediction_result(
    transaction: Dict,
    prediction: bool,
    actual_label: bool,
    predict_proba: float,
    prediction_time: Optional[str] = None
) -> None:
    """
    Store a prediction result in the appropriate database table.
    
    This function automatically:
    - Routes correct predictions (TP, TN) to correct_predictions table
    - Routes incorrect predictions (FP, FN) to incorrect_predictions table with predict_proba
    
    Args:
        transaction: Transaction data dictionary (all fields from Transaction schema)
        prediction: Model prediction (True/False for is_fraud)
        actual_label: Actual label from ground truth (True/False for is_fraud)
        predict_proba: Probability score from model (0.0 to 1.0)
        prediction_time: Optional ISO timestamp (defaults to current UTC time)
    
    Example:
        >>> transaction = {
        ...     "transac_type": "CASH_OUT",
        ...     "amount": 1000.0,
        ...     "src_bal": 5000.0,
        ...     "src_new_bal": 4000.0
        ... }
        >>> store_prediction_result(
        ...     transaction=transaction,
        ...     prediction=True,
        ...     actual_label=True,  # TP case
        ...     predict_proba=0.85
        ... )
    """
    # Ensure database is initialized
    try:
        init_prediction_db()
    except Exception:
        pass  # Database might already exist
    
    save_prediction(
        transaction=transaction,
        prediction=prediction,
        actual_label=actual_label,
        predict_proba=predict_proba,
        prediction_time=prediction_time
    )


def get_model_performance_stats() -> Dict:
    """
    Get performance statistics from stored predictions.
    
    Returns:
        Dictionary with:
        - total_predictions: Total number of predictions
        - correct_predictions: Number of correct predictions
        - incorrect_predictions: Number of incorrect predictions
        - true_positives: TP count
        - true_negatives: TN count
        - false_positives: FP count
        - false_negatives: FN count
        - accuracy: Accuracy percentage
    """
    return get_prediction_stats()

