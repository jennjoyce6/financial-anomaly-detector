import sqlite3
import pandas as pd
from fastapi import APIRouter, HTTPException
from api.schemas import (
    TransactionRequest, DetectionResponse,
    BatchRequest, BatchResponse, StatsResponse
)
from models.anomaly_detector import AnomalyDetector

router    = APIRouter()
detector  = AnomalyDetector()

def _load_baseline():
    try:
        conn = sqlite3.connect('data/transactions.db')
        df   = pd.read_sql('SELECT amount FROM transactions', conn)
        conn.close()
        return {
            'mean': df['amount'].mean(),
            'std':  df['amount'].std(),
            'q1':   df['amount'].quantile(0.25),
            'q3':   df['amount'].quantile(0.75),
        }
    except Exception:
        return None

# Simple in-memory stats tracker
_baseline = _load_baseline()
_stats = {"total_analyzed": 0, "total_flagged": 0, "high_risk_count": 0}


def _analyze_transaction(txn: TransactionRequest) -> DetectionResponse:
    """Run all three detectors on a single transaction."""

    df = pd.DataFrame([{
        "amount":    txn.amount,
        "timestamp": pd.Timestamp.now().replace(hour=txn.hour),
        "txn_type":  txn.txn_type,
    }])

    results  = detector.run_all(df)
    row      = results.iloc[0]

    # Use baseline stats for z-score if available
    # This fixes the cold start problem for single-transaction analysis
    z_score = float(row['z_score']) if pd.notna(row['z_score']) else 0.0
    z_flag  = False

    if _baseline:
        z_score = round(
            (txn.amount - _baseline['mean']) / _baseline['std'], 3
        )
        z_flag = abs(z_score) > 3.0

        iqr    = _baseline['q3'] - _baseline['q1']
        iqr_flag = (
            txn.amount < (_baseline['q1'] - 1.5 * iqr) or
            txn.amount > (_baseline['q3'] + 1.5 * iqr)
        )
    else:
        z_flag   = bool(row['z_flag'])
        iqr_flag = bool(row['iqr_flag'])

    iso_flag = bool(row['iso_flag'])

    # Recalculate consensus with corrected flags
    flag_count    = int(z_flag) + int(iqr_flag) + int(iso_flag)
    is_anomaly    = flag_count >= 1
    anomaly_score = round(flag_count / 3, 2)

    # Risk tier
    if anomaly_score >= 0.67:
        risk_level = "High"
    elif anomaly_score >= 0.34:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Human readable flags
    flags = []
    if z_flag:
        flags.append("large_amount")
    if iqr_flag:
        flags.append("statistical_outlier")
    if iso_flag:
        flags.append("multivariate_anomaly")
    if txn.hour < 6 or txn.hour > 22:
        flags.append("off_hours")
    if 9990 <= txn.amount <= 9999:
        flags.append("structuring_pattern")

    return DetectionResponse(
        txn_id        = txn.txn_id,
        is_anomaly    = is_anomaly,
        risk_level    = risk_level,
        anomaly_score = anomaly_score,
        z_score       = z_score,
        iso_score     = float(row['iso_score']),
        flags         = flags,
    )


@router.post("/detect", response_model=DetectionResponse)
def detect_single(txn: TransactionRequest):
    """Analyze a single financial transaction for anomalies."""
    result = _analyze_transaction(txn)

    # Update stats
    _stats["total_analyzed"] += 1
    if result.is_anomaly:
        _stats["total_flagged"] += 1
    if result.risk_level == "High":
        _stats["high_risk_count"] += 1

    return result


@router.post("/detect/batch", response_model=BatchResponse)
def detect_batch(batch: BatchRequest):
    """Analyze a batch of up to 1,000 transactions."""
    if not batch.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")

    results = [_analyze_transaction(txn) for txn in batch.transactions]
    flagged = [r for r in results if r.is_anomaly]

    # Update stats
    _stats["total_analyzed"] += len(results)
    _stats["total_flagged"]  += len(flagged)
    _stats["high_risk_count"] += sum(
        1 for r in flagged if r.risk_level == "High"
    )

    return BatchResponse(
        total=len(results),
        flagged=len(flagged),
        results=results,
    )


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    """Return aggregate detection statistics for this session."""
    total    = _stats["total_analyzed"]
    flagged  = _stats["total_flagged"]
    rate     = round(flagged / total * 100, 2) if total > 0 else 0.0

    return StatsResponse(
        total_analyzed  = total,
        total_flagged   = flagged,
        high_risk_count = _stats["high_risk_count"],
        detection_rate  = rate,
    )