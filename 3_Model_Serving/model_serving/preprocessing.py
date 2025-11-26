from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import os

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from enum import Enum as _Enum
import logging

from .schemas import TRANSAC_TYPE, ALLOWED_TRANSAC_TYPES

FEATURE_COLUMNS = [
    "type",
    "amount",
    'oldbalanceOrg', 
    'newbalanceOrig', 
    'oldbalanceDest', 
    'newbalanceDest'
]

NUMERIC_FEATURES = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

logger = logging.getLogger(__name__)

def parse_time_features(time_ind: str) -> Dict[str, Any]:
    try:
        dt = datetime.fromisoformat(time_ind)
    except Exception:
        try:
            dt = datetime.strptime(time_ind, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return {"hour": None, "day_of_week": None, "parsed": False}

    return {"hour": dt.hour, "day_of_week": dt.weekday(), "parsed": True}


def load_preprocessing_artifacts(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load persisted preprocessing artifacts from disk."""
    if path is None:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(parent_dir, "models", "preprocessing_artifacts.joblib")

    try:
        artifacts = joblib.load(path)
        return artifacts
    except Exception:
        return None


def transform_transaction(transaction: Dict[str, Any], artifacts: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Convert a single transaction dict into a feature-aligned dataframe."""
    feature_row = _transaction_to_feature_row(transaction)
    feature_df = pd.DataFrame([feature_row])
    transformed, _ = prepare_feature_frame(feature_df, artifacts=artifacts, fit=False)
    return transformed


def transaction_to_features(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy helper that extracts a subset of scalar features from a transaction dict."""
    features: Dict[str, Any] = {}
    features["amount"] = transaction.get("amount")

    time_ind = transaction.get("step")
    if time_ind is not None:
        time_feats = parse_time_features(str(time_ind))
        features.update(time_feats)
    else:
        features.update({"hour": None, "day_of_week": None, "parsed": False})

    features["src_acc"] = transaction.get("src_acc")
    features["dst_acc"] = transaction.get("dst_acc")

    return features


def build_feature_source_from_master(df: pd.DataFrame) -> pd.DataFrame:
    """Map the master transaction table schema into the model feature schema."""
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    work = df.copy()
    work.columns = [str(c) for c in work.columns]

    def _coalesce(*cols):
        for col in cols:
            if col in work.columns:
                return work[col]
        return pd.Series([None] * len(work))

    feature_df = pd.DataFrame()
    # Align output columns exactly with FEATURE_COLUMNS expected by the model
    feature_df["type"] = _coalesce("type", "transac_type")
    feature_df["amount"] = _coalesce("amount")
    feature_df["oldbalanceOrg"] = _coalesce("oldbalanceOrg", "src_bal")
    feature_df["newbalanceOrig"] = _coalesce("newbalanceOrig", "src_new_bal")
    feature_df["oldbalanceDest"] = _coalesce("oldbalanceDest", "dst_bal")
    feature_df["newbalanceDest"] = _coalesce("newbalanceDest", "dst_new_bal")

    # Return columns in the exact order/name expected downstream
    return feature_df[FEATURE_COLUMNS]


def prepare_feature_frame(
    df: pd.DataFrame,
    artifacts: Optional[Dict[str, Any]] = None,
    fit: bool = False,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Apply one-hot encoding + scaling on the core feature frame.

    When ``fit`` is True, new artifacts (scaler, column order) are created.
    Otherwise the provided artifacts are used to align the columns and scale.
    """
    working = df.copy()

    if working.empty:
        return working, artifacts

    # Expect the column named 'type' (matches FEATURE_COLUMNS)
    if "type" not in working.columns:
        raise ValueError("Expected 'type' column in feature dataframe")

    working["type"] = working["type"].apply(_normalize_transac_type)
    working["type"] = pd.Categorical(
        working["type"], categories=ALLOWED_TRANSAC_TYPES
    )
    working = pd.get_dummies(working, columns=["type"], drop_first=True)

    if fit:
        train_cols = list(working.columns)
        num_cols = [c for c in NUMERIC_FEATURES if c in working.columns]
        scaler = None
        if num_cols:
            scaler = StandardScaler()
            working[num_cols] = scaler.fit_transform(working[num_cols].astype(float))
        artifacts = {
            "train_cols": train_cols,
            "numerical_cols": num_cols,
            "scaler": scaler,
        }
        return working, artifacts

    if artifacts is None or not artifacts.get("train_cols"):
        raise ValueError("Preprocessing artifacts are required for inference")

    train_cols = list(artifacts["train_cols"])
    for col in train_cols:
        if col not in working.columns:
            working[col] = 0
    extra_cols = set(working.columns) - set(train_cols)
    if extra_cols:
        working = working.drop(columns=list(extra_cols))
    working = working[train_cols]

    scaler = artifacts.get("scaler")
    if scaler is not None:
        # Use scaler's own feature names if available (most reliable)
        if hasattr(scaler, "feature_names_in_"):
            scaler_cols = list(scaler.feature_names_in_)
        else:
            # Fallback to numerical_cols from artifacts
            scaler_cols = list(artifacts.get("numerical_cols", []))

        if not scaler_cols:
            logger.warning("No scaler feature columns found; skipping scaler transform")
        else:
            # Mapping between canonical names (in current data) and legacy names
            canonical_to_legacy = {
                "oldbalanceOrg": "src_bal",
                "newbalanceOrig": "src_new_bal",
                "oldbalanceDest": "dst_bal",
                "newbalanceDest": "dst_new_bal",
            }
            legacy_to_canonical = {v: k for k, v in canonical_to_legacy.items()}

            # For each column the scaler needs, ensure it's present in working
            for expected in scaler_cols:
                if expected in working.columns:
                    continue
                # If expected is a legacy name (e.g. src_bal), try to get from canonical name
                if expected in legacy_to_canonical:
                    canonical_name = legacy_to_canonical[expected]
                    if canonical_name in working.columns:
                        working[expected] = working[canonical_name]
                        logger.debug("Mapped %s -> %s for scaler", canonical_name, expected)
                        continue
                # If expected is a canonical name, try to get from legacy name
                if expected in canonical_to_legacy:
                    legacy_name = canonical_to_legacy[expected]
                    if legacy_name in working.columns:
                        working[expected] = working[legacy_name]
                        logger.debug("Mapped %s -> %s for scaler", legacy_name, expected)
                        continue

            all_present = all(c in working.columns for c in scaler_cols)

            if all_present:
                # Transform using the scaler's expected column order
                working[scaler_cols] = scaler.transform(working[scaler_cols].astype(float))
            else:
                logger.warning(
                    "Scaler transform skipped because not all expected numeric columns are present: %s",
                    scaler_cols,
                )
    else:
        for col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    return working, artifacts


def _transaction_to_feature_row(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a transaction dict into the expected feature row schema."""
    row: Dict[str, Any] = {}
    t_type = transaction.get("type") or transaction.get("transac_type")
    if isinstance(t_type, _Enum):
        t_type = t_type.value
    row["type"] = _normalize_transac_type(t_type)
    row["amount"] = transaction.get("amount")
    row["oldbalanceOrg"] = transaction.get("oldbalanceOrg", transaction.get("src_bal"))
    row["newbalanceOrig"] = transaction.get("newbalanceOrig", transaction.get("src_new_bal"))
    row["oldbalanceDest"] = transaction.get("oldbalanceDest", transaction.get("dst_bal"))
    row["newbalanceDest"] = transaction.get("newbalanceDest", transaction.get("dst_new_bal"))
    return row


def _normalize_transac_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if normalized and normalized not in TRANSAC_TYPE.__members__:
        raise ValueError(f"Invalid transac_type '{value}'; allowed: {list(TRANSAC_TYPE)}")
    return normalized
