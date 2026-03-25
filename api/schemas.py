from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class TransactionType(str, Enum):
    wire_transfer  = "wire_transfer"
    ach_debit      = "ach_debit"
    vendor_payment = "vendor_payment"
    payroll        = "payroll"
    intl_payment   = "intl_payment"


class TransactionRequest(BaseModel):
    """Single transaction to analyze."""
    txn_id:   str            = Field(..., example="TXN-88421")
    amount:   float          = Field(..., gt=0, example=48000.00)
    txn_type: TransactionType = Field(..., example="wire_transfer")
    hour:     int            = Field(..., ge=0, le=23, example=23)

    class Config:
        use_enum_values = True


class DetectionResponse(BaseModel):
    """Result for a single analyzed transaction."""
    txn_id:        str
    is_anomaly:    bool
    risk_level:    str
    anomaly_score: float
    z_score:       float
    iso_score:     float
    flags:         List[str]


class BatchRequest(BaseModel):
    """Up to 1,000 transactions for batch analysis."""
    transactions: List[TransactionRequest] = Field(
        ..., max_length=1000
    )


class BatchResponse(BaseModel):
    """Results for a batch of analyzed transactions."""
    total:   int
    flagged: int
    results: List[DetectionResponse]


class StatsResponse(BaseModel):
    """Aggregate detection statistics."""
    total_analyzed:   int
    total_flagged:    int
    high_risk_count:  int
    detection_rate:   float