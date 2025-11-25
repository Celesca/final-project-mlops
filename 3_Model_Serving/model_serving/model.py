from typing import Tuple, Dict, Any, Optional
from pydantic import BaseModel
import os
import joblib
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FraudDetectionModel:
    def __init__(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            # Default: look for xgb_model.joblib in ../models/ (parent of model_serving/)
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(parent_dir, "models", "xgb_model.joblib")
        
        self.model_path = model_path
        self.model = None
        self._has_proba = False
        self._is_xgb_booster = False

        # Auto-load if path exists
        if model_path and os.path.exists(model_path):
            try:
                self.load(model_path)
            except Exception:
                # keep fallback behavior
                self.model = None

    def load(self, path: str) -> None:

        obj = joblib.load(path)
        self.model = obj
        # detect capabilities
        self._has_proba = hasattr(self.model, "predict_proba")

        # Detect XGBoost native Booster (joblib may load a xgboost.Booster)
        try:
            import xgboost as xgb

            self._is_xgb_booster = isinstance(self.model, xgb.Booster)
        except Exception:
            # xgboost not installed or the model is not a Booster
            self._is_xgb_booster = False

        self.model_path = path

    def _to_dataframe(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Normalize input features to a single-row DataFrame."""
        if isinstance(features, pd.DataFrame):
            return features.reset_index(drop=True)
        # assume mapping-like
        return pd.DataFrame([features])

    def predict(self, features: Dict[str, Any]) -> Tuple[bool, float]:
        # Return (is_fraud, probability).
        
        if self.model is None:
            raise RuntimeError("No model loaded")

        logger.debug("Running model-based prediction")

        df = self._to_dataframe(features)

        # 1) If model exposes predict_proba (sklearn-like or XGB sklearn API)
        if self._has_proba:
            probs = self.model.predict_proba(df)
            if isinstance(probs, (np.ndarray, list)) and getattr(probs, "ndim", 1) == 2 and probs.shape[1] >= 2:
                prob = float(probs[0, 1])
            else:
                prob = float(np.asarray(probs).ravel()[0])
            is_fraud = prob >= 0.5
            return bool(is_fraud), float(prob)

        # 2) If it's an XGBoost native Booster, convert to DMatrix and predict
        if self._is_xgb_booster:
            try:
                import xgboost as xgb

                dmat = xgb.DMatrix(df.values, feature_names=list(df.columns))
                preds = self.model.predict(dmat)
                prob = float(np.asarray(preds).ravel()[0])
                is_fraud = prob >= 0.5
                return bool(is_fraud), float(prob)
            except Exception:
                logger.exception("XGBoost Booster prediction failed")
                raise

        # 3) Generic predict: treat numeric outputs as probabilities or class labels
        preds = self.model.predict(df)
        if isinstance(preds, (np.ndarray, list, tuple)):
            val = np.asarray(preds).ravel()[0]
            try:
                prob = float(val)
            except Exception:
                try:
                    prob = 1.0 if int(val) == 1 else 0.0
                except Exception:
                    raise RuntimeError("Model returned unsupported prediction type")
            is_fraud = prob >= 0.5
            return bool(is_fraud), float(prob)

        # scalar-like prediction
        try:
            prob = float(preds)
            is_fraud = prob >= 0.5
            return bool(is_fraud), float(prob)
        except Exception:
            raise RuntimeError("Model returned unsupported prediction type")

    def predict_from_transaction(self, transaction: BaseModel) -> Tuple[bool, float]:
        """Helper to accept a pydantic transaction model or raw dict."""
        if hasattr(transaction, "dict"):
            data = transaction.dict()
        else:
            data = dict(transaction)
        return self.predict(data)
