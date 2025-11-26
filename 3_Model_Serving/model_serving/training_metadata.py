"""
Utility functions for storing and retrieving training metadata (e.g., training date).
"""
import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime


# Path to store training metadata (relative to models directory)
MODELS_DIR = Path(__file__).parent.parent / "models"
TRAINING_METADATA_PATH = MODELS_DIR / "training_metadata.json"


def save_training_date(training_date: str) -> None:
    """
    Save the training date to a JSON file.
    
    Args:
        training_date: Date string in YYYY-MM-DD format representing the last date
                       of data used for training
    """
    # Create directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing metadata or create new
    metadata = {}
    if TRAINING_METADATA_PATH.exists():
        try:
            with open(TRAINING_METADATA_PATH, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
            metadata = {}
    
    # Update training date
    metadata['last_training_date'] = training_date
    metadata['last_updated'] = datetime.utcnow().isoformat()
    
    # Save to file
    with open(TRAINING_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved training date: {training_date} to {TRAINING_METADATA_PATH}")


def get_training_date() -> Optional[str]:
    """
    Retrieve the last training date from the metadata file.
    
    Returns:
        Date string in YYYY-MM-DD format, or None if not found
    """
    if not TRAINING_METADATA_PATH.exists():
        return None
    
    try:
        with open(TRAINING_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        return metadata.get('last_training_date')
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ Error reading training metadata: {e}")
        return None

