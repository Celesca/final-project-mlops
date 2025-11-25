from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import date, datetime
from enum import Enum

class TRANSAC_TYPE(str, Enum):
    CASH_OUT = "CASH_OUT"
    PAYMENT = "PAYMENT"
    CASH_IN = "CASH_IN"
    TRANSFER = "TRANSFER"
    DEBIT = "DEBIT"


# Public list of allowed transaction type strings (used by preprocessing/pandas categories)
ALLOWED_TRANSAC_TYPES = list(TRANSAC_TYPE.__members__.keys())


class Transaction(BaseModel):
    # make time_ind optional for clients that don't provide timestamps
    time_ind: Optional[int] = Field(None, description="Transaction timestamp or time index")
    transac_type: TRANSAC_TYPE = Field(..., description="Transaction type / channel")
    amount: float = Field(..., gt=0, description="Transaction amount")
    # source / destination accounts are optional for some synthetic/test payloads
    src_acc: Optional[str] = Field(None, description="Source account identifier")
    src_bal: Optional[float] = Field(None, description="Source account balance before transaction")
    src_new_bal: Optional[float] = Field(None, description="Source account balance after transaction")
    dst_acc: Optional[str] = Field(None, description="Destination account identifier")
    dst_bal: Optional[float] = Field(None, description="Destination account balance before transaction")
    dst_new_bal: Optional[float] = Field(None, description="Destination account balance after transaction")

    class Config:
        schema_extra = {
            "example": {
                "transac_type": "CASH_OUT",
                "amount": 181.00,
                "src_bal": 181.0,
                "src_new_bal": 0,
                "dst_bal": 21182.0,
                "dst_new_bal": 0
            }
        }


class PredictionResponse(BaseModel):
    is_fraud: bool = Field(..., description="Fraud prediction (True/False)")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Fraud probability score")
    prediction_time: str = Field(..., description="ISO timestamp of prediction")


class FraudTransaction(BaseModel):
    id: int
    transaction_data: Dict
    fraud_probability: float
    prediction_time: str


class LoadModelRequest(BaseModel):
    """Request model for loading a model artifact from a filesystem path.

    The `model_path` should point to a file that `joblib.load` can read
    (for example a model saved by MLflow or joblib). This endpoint does not
    attempt to download from remote MLflow registries — pass a local path.
    """
    model_path: str


class MasterTransaction(BaseModel):
    """Schema mirroring the master `all_transactions` table columns."""

    step: Optional[int] = Field(None, description="Time step or transaction index")
    type: Optional[str] = Field(None, description="Raw transaction type")
    transac_type: Optional[str] = Field(None, description="Normalized transaction type")
    amount: float = Field(..., description="Transaction amount")
    nameOrig: Optional[str] = Field(None, description="Source account identifier")
    src_acc: Optional[str] = Field(None, description="Alias for source account")
    oldbalanceOrg: Optional[float] = Field(None, description="Source balance before transaction")
    src_bal: Optional[float] = Field(None, description="Alias for source balance before transaction")
    newbalanceOrig: Optional[float] = Field(None, description="Source balance after transaction")
    src_new_bal: Optional[float] = Field(None, description="Alias for source balance after transaction")
    nameDest: Optional[str] = Field(None, description="Destination account identifier")
    dst_acc: Optional[str] = Field(None, description="Alias for destination account")
    oldbalanceDest: Optional[float] = Field(None, description="Destination balance before transaction")
    dst_bal: Optional[float] = Field(None, description="Alias for destination balance before transaction")
    newbalanceDest: Optional[float] = Field(None, description="Destination balance after transaction")
    dst_new_bal: Optional[float] = Field(None, description="Alias for destination balance after transaction")
    isFraud: Optional[int] = Field(None, description="Ground-truth fraud label")
    isFlaggedFraud: Optional[int] = Field(None, description="Flagged indicator from source system")
    ingest_date: Optional[date] = Field(None, description="Ingestion date for master table entry")
    source_file: Optional[str] = Field(None, description="Source file identifier")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")

    class Config:
        extra = "allow"


class TrainRequest(BaseModel):
    """Empty payload for /train (kept for forward compatibility)."""

    class Config:
        extra = "forbid"


class TrainResponse(BaseModel):
    run_id: str
    val_auc: float
    val_accuracy: float
    promoted: bool
    model_path: Optional[str]
    message: str


class PredictRequest(BaseModel):
    new_data: List[MasterTransaction] = Field(
        ..., description="Batch of transactions to score"
    )


class EnrichedTransaction(MasterTransaction):
    predict_proba: float = Field(..., ge=0.0, le=1.0, description="Fraud probability")
    prediction: int = Field(..., description="Binary fraud prediction (1=fraud)")
