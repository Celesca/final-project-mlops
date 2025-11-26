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

