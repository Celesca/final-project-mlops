"""
Drift detection utility functions using Evidently.
"""
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def is_drift_critical(
    reference_data: pd.DataFrame, 
    current_data: pd.DataFrame, 
    threshold: float = 0.3
) -> bool:
    """
    Returns True if the share of drifted columns exceeds the provided threshold.
    
    Args:
        reference_data: Baseline data (training set).
        current_data: Production/Inference data.
        threshold: Float between 0 and 1 (e.g., 0.3 for 30%).
    
    Returns:
        bool: True if drift exceeds threshold, False otherwise.
    """
    # Setup and Run drift report
    # We use the Preset for simplicity, as it covers all columns automatically
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=current_data)
    results_dict = drift_report.metrics[0].dict()
    drift_share = results_dict['drift_share']
    
    # Compare and Return
    # If 30% drifted and threshold is 0.2, return True (Alarm triggered)
    return drift_share > threshold

