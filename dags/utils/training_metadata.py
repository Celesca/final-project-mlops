"""
Utility functions for storing and retrieving training metadata (e.g., training date).
"""
import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime


# Path to store training metadata
TRAINING_METADATA_PATH = os.getenv(
    "TRAINING_METADATA_PATH",
    os.path.join(os.path.dirname(__file__), "../../3_Model_Serving/models/training_metadata.json")
)


def get_training_metadata_path() -> str:
    """Get the absolute path to the training metadata file."""
    return os.path.abspath(TRAINING_METADATA_PATH)


def save_training_date(training_date: str) -> None:
    """
    Save the training date to a JSON file.
    
    Args:
        training_date: Date string in YYYY-MM-DD format representing the last date
                       of data used for training
    """
    metadata_path = get_training_metadata_path()
    metadata_dir = os.path.dirname(metadata_path)
    
    # Create directory if it doesn't exist
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Load existing metadata or create new
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
            metadata = {}
    
    # Update training date
    metadata['last_training_date'] = training_date
    metadata['last_updated'] = datetime.utcnow().isoformat()
    
    # Save to file
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved training date: {training_date} to {metadata_path}")


def get_training_date() -> Optional[str]:
    """
    Retrieve the last training date from the metadata file.
    
    Returns:
        Date string in YYYY-MM-DD format, or None if not found
    """
    metadata_path = get_training_metadata_path()
    
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata.get('last_training_date')
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ Error reading training metadata: {e}")
        return None


def save_baseline_date(baseline_date: str) -> None:
    """
    Save the baseline date to a JSON file.
    This is used for drift detection when no training date exists yet.
    
    Args:
        baseline_date: Date string in YYYY-MM-DD format representing the baseline
                       date for drift detection
    """
    metadata_path = get_training_metadata_path()
    metadata_dir = os.path.dirname(metadata_path)
    
    # Create directory if it doesn't exist
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Load existing metadata or create new
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
            metadata = {}
    
    # Update baseline date
    metadata['baseline_date'] = baseline_date
    metadata['baseline_last_updated'] = datetime.utcnow().isoformat()
    
    # Save to file
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved baseline date: {baseline_date} to {metadata_path}")


def get_baseline_date() -> Optional[str]:
    """
    Retrieve the baseline date from the metadata file.
    This is used for drift detection when no training date exists yet.
    
    Returns:
        Date string in YYYY-MM-DD format, or None if not found
    """
    metadata_path = get_training_metadata_path()
    
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata.get('baseline_date')
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ Error reading training metadata: {e}")
        return None


def clear_baseline_date() -> None:
    """
    Clear/remove the baseline date from metadata.
    Useful for resetting drift detection when starting a new simulation.
    """
    metadata_path = get_training_metadata_path()
    
    if not os.path.exists(metadata_path):
        print("ℹ️ No metadata file found. Nothing to clear.")
        return
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ Error reading training metadata: {e}")
        return
    
    if 'baseline_date' in metadata:
        del metadata['baseline_date']
        if 'baseline_last_updated' in metadata:
            del metadata['baseline_last_updated']
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Cleared baseline date from {metadata_path}")
    else:
        print("ℹ️ No baseline date found in metadata. Nothing to clear.")


def get_next_simulation_day() -> int:
    """
    Get the next simulation day to process and increment the counter.
    This allows the simulation to progress independently of the actual execution date.
    
    Returns:
        int: The simulation day to process (0-indexed)
    """
    metadata_path = get_training_metadata_path()
    metadata_dir = os.path.dirname(metadata_path)
    
    # Create directory if it doesn't exist
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Load existing metadata or create new
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
            metadata = {}
    
    # Get current simulation day (default to -1 so first run is day 0)
    current_day = metadata.get('last_simulation_day', -1)
    
    # Increment for next run
    next_day = current_day + 1
    metadata['last_simulation_day'] = next_day
    metadata['simulation_last_updated'] = datetime.utcnow().isoformat()
    
    # Save to file
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📅 Next simulation day: {next_day} (incremented from {current_day})")
    return next_day


def get_simulation_date(simulation_day: int) -> str:
    """
    Convert simulation day to a simulated date string.
    Uses DAG_START_DATE as the base and adds the simulation day.
    
    Args:
        simulation_day: The simulation day (0-indexed)
    
    Returns:
        Date string in YYYY-MM-DD format
    """
    from dags.config import DAG_START_DATE
    from datetime import timedelta
    
    base_date = datetime.strptime(DAG_START_DATE, "%Y-%m-%d")
    simulated_date = base_date + timedelta(days=simulation_day)
    return simulated_date.strftime("%Y-%m-%d")


def reset_simulation_day() -> None:
    """
    Reset the simulation day counter to -1 (so next run will be day 0).
    Useful for restarting the simulation from the beginning.
    """
    metadata_path = get_training_metadata_path()
    metadata_dir = os.path.dirname(metadata_path)
    
    # Create directory if it doesn't exist
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Load existing metadata or create new
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
            metadata = {}
    
    # Reset simulation day to -1
    metadata['last_simulation_day'] = -1
    metadata['simulation_reset_at'] = datetime.utcnow().isoformat()
    
    # Save to file
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Reset simulation day counter to -1 (next run will be day 0)")


def get_current_simulation_day() -> int:
    """
    Get the current simulation day without incrementing.
    
    Returns:
        int: The current simulation day (or -1 if never run)
    """
    metadata_path = get_training_metadata_path()
    
    if not os.path.exists(metadata_path):
        return -1
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata.get('last_simulation_day', -1)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ Error reading training metadata: {e}")
        return -1

