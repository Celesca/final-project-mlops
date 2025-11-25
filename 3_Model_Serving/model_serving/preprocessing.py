from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import os

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from enum import Enum as _Enum

from .schemas import TRANSAC_TYPE, ALLOWED_TRANSAC_TYPES

FEATURE_COLUMNS = [
    "transac_type",
    "amount",
    "src_bal",
    "src_new_bal",
    "dst_bal",
    "dst_new_bal",
]

NUMERIC_FEATURES = ["amount", "src_bal", "src_new_bal", "dst_bal", "dst_new_bal"]


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

    time_ind = transaction.get("time_ind")
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
    feature_df["transac_type"] = _coalesce("transac_type", "type")
    feature_df["amount"] = _coalesce("amount")
    feature_df["src_bal"] = _coalesce("src_bal", "oldbalanceOrg")
    feature_df["src_new_bal"] = _coalesce("src_new_bal", "newbalanceOrig")
    feature_df["dst_bal"] = _coalesce("dst_bal", "oldbalanceDest")
    feature_df["dst_new_bal"] = _coalesce("dst_new_bal", "newbalanceDest")

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

    if "transac_type" not in working.columns:
        raise ValueError("Expected 'transac_type' column in feature dataframe")

    working["transac_type"] = working["transac_type"].apply(_normalize_transac_type)
    working["transac_type"] = pd.Categorical(
        working["transac_type"], categories=ALLOWED_TRANSAC_TYPES
    )
    working = pd.get_dummies(working, columns=["transac_type"], drop_first=True)

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
        num_cols = artifacts.get("numerical_cols", [])
        num_cols = [c for c in num_cols if c in working.columns]
        if num_cols:
            working[num_cols] = scaler.transform(working[num_cols].astype(float))
    else:
        for col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    return working, artifacts


def _transaction_to_feature_row(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a transaction dict into the expected feature row schema."""
    row: Dict[str, Any] = {}
    t_type = transaction.get("transac_type") or transaction.get("type")
    if isinstance(t_type, _Enum):
        t_type = t_type.value
    row["transac_type"] = _normalize_transac_type(t_type)
    row["amount"] = transaction.get("amount")
    row["src_bal"] = transaction.get("src_bal", transaction.get("oldbalanceOrg"))
    row["src_new_bal"] = transaction.get("src_new_bal", transaction.get("newbalanceOrig"))
    row["dst_bal"] = transaction.get("dst_bal", transaction.get("oldbalanceDest"))
    row["dst_new_bal"] = transaction.get("dst_new_bal", transaction.get("newbalanceDest"))
    return row


def _normalize_transac_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if normalized and normalized not in TRANSAC_TYPE.__members__:
        raise ValueError(f"Invalid transac_type '{value}'; allowed: {list(TRANSAC_TYPE)}")
    return normalized
