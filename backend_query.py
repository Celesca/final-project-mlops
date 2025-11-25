from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dags.utils import database as db

app = FastAPI(title="Fraud Predictions Query API")


class UpdatePredictionRequest(BaseModel):
    transaction_id: int
    actual_label: bool


def _clean_transaction_data(txn: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive/undeclared fields from transaction_data per request.

    We drop: `isFraud`, `isFlaggedFraud`, `ingest_date`, `source_file`.
    """
    if not txn:
        return {}
    return {k: v for k, v in txn.items() if k not in {"isFraud", "isFlaggedFraud", "ingest_date", "source_file"}}


@app.get("/GET/predictions")
def get_predictions(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return joined predictions from correct and incorrect prediction tables.

    Each record includes prediction metadata and cleaned `transaction_data`.
    """
    correct = db.get_correct_predictions(limit)
    incorrect = db.get_incorrect_predictions(limit)

    combined: List[Dict[str, Any]] = []

    for r in correct:
        tx = _clean_transaction_data(r.get("transaction_data") or {})
        combined.append(
            {
                "id": r.get("id"),
                "transaction_id": r.get("transaction_id"),
                "prediction": r.get("prediction"),
                "actual_label": r.get("actual_label"),
                "prediction_time": r.get("prediction_time"),
                "created_at": r.get("created_at"),
                "transaction_data": tx,
                "source": "correct",
            }
        )

    for r in incorrect:
        tx = _clean_transaction_data(r.get("transaction_data") or {})
        combined.append(
            {
                "id": r.get("id"),
                "transaction_id": r.get("transaction_id"),
                "prediction": r.get("prediction"),
                "actual_label": r.get("actual_label"),
                "predict_proba": r.get("predict_proba"),
                "prediction_time": r.get("prediction_time"),
                "created_at": r.get("created_at"),
                "transaction_data": tx,
                "source": "incorrect",
            }
        )

    # sort by prediction_time (descending) when available
    combined.sort(key=lambda x: x.get("prediction_time") or "", reverse=True)
    return combined


@app.get("/get/non_frauds")
def get_non_frauds(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return transaction details coming only from `correct_predictions`.

    Transaction details are cleaned to remove sensitive/hidden fields.
    """
    correct = db.get_correct_predictions(limit)
    results: List[Dict[str, Any]] = []
    for r in correct:
        tx = _clean_transaction_data(r.get("transaction_data") or {})
        results.append(
            {
                "transaction_id": r.get("transaction_id"),
                "prediction": r.get("prediction"),
                "actual_label": r.get("actual_label"),
                "prediction_time": r.get("prediction_time"),
                "transaction_data": tx,
            }
        )
    return results


@app.get("/GET/frauds")
def get_frauds(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return transaction details coming only from `incorrect_predictions` (FP and FN).

    Includes `predict_proba` and cleaned transaction_data.
    """
    incorrect = db.get_incorrect_predictions(limit)
    results: List[Dict[str, Any]] = []
    for r in incorrect:
        tx = _clean_transaction_data(r.get("transaction_data") or {})
        results.append(
            {
                "transaction_id": r.get("transaction_id"),
                "prediction": r.get("prediction"),
                "actual_label": r.get("actual_label"),
                "predict_proba": r.get("predict_proba"),
                "prediction_time": r.get("prediction_time"),
                "transaction_data": tx,
            }
        )
    return results


@app.put("/PUT/predictions")
def update_prediction_label(payload: UpdatePredictionRequest) -> Dict[str, int]:
    """Update the `actual_label` for a prediction record by `transaction_id`.

    This updates whichever table(s) contain the `transaction_id` (correct_predictions
    and/or incorrect_predictions). It does not move records between tables.
    """
    tid = payload.transaction_id
    actual_int = 1 if payload.actual_label else 0

    updated_correct = 0
    updated_incorrect = 0

    # Use the context manager from database.py to run two updates in a single transaction
    with db.get_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE correct_predictions SET actual_label = %s WHERE transaction_id = %s",
            (actual_int, tid),
        )
        updated_correct = cur.rowcount

        cur.execute(
            "UPDATE incorrect_predictions SET actual_label = %s WHERE transaction_id = %s",
            (actual_int, tid),
        )
        updated_incorrect = cur.rowcount

    if (updated_correct + updated_incorrect) == 0:
        raise HTTPException(status_code=404, detail=f"No prediction record found for transaction_id={tid}")

    return {"updated_correct": updated_correct, "updated_incorrect": updated_incorrect}
