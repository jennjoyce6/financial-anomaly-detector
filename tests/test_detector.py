import pytest
import pandas as pd
import numpy as np
from models.anomaly_detector import AnomalyDetector
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def sample_df():
    """Minimal transaction DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'amount':    [100, 200, 150, 50000, 120, 180],  # 50000 is the anomaly
        'timestamp': pd.date_range('2026-01-01', periods=6, freq='h'),
        'txn_type':  ['wire_transfer'] * 6,
    })

def test_zscore_flags_outlier(sample_df):
    detector = AnomalyDetector()
    df = detector.add_features(sample_df)
    _, flags = detector.zscore_detection(df, threshold=2.0)
    assert flags.iloc[3] == True   # 50000 should be flagged

def test_iqr_flags_outlier(sample_df):
    detector = AnomalyDetector()
    df = detector.add_features(sample_df)
    flags = detector.iqr_detection(df)
    assert flags.iloc[3] == True

def test_run_all_returns_expected_columns(sample_df):
    detector = AnomalyDetector()
    result = detector.run_all(sample_df)
    for col in ['z_score', 'iqr_flag', 'iso_flag',
                'anomaly_score', 'predicted_anomaly', 'risk_level']:
        assert col in result.columns

def test_anomaly_score_range(sample_df):
    detector = AnomalyDetector()
    result = detector.run_all(sample_df)
    assert result['anomaly_score'].between(0, 1).all()

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_detect_normal_transaction():
    response = client.post("/detect", json={
        "txn_id":   "TXN-TEST-001",
        "amount":   450.00,
        "txn_type": "ach_debit",
        "hour":     14
    })
    assert response.status_code == 200
    data = response.json()
    assert data["txn_id"] == "TXN-TEST-001"
    assert "is_anomaly" in data
    assert "risk_level" in data
    assert "flags" in data

def test_detect_suspicious_transaction():
    response = client.post("/detect", json={
        "txn_id":   "TXN-TEST-002",
        "amount":   48000.00,
        "txn_type": "wire_transfer",
        "hour":     23
    })
    assert response.status_code == 200
    data = response.json()
    assert data["is_anomaly"] == True
    assert data["risk_level"] == "High"
    assert "off_hours" in data["flags"]

def test_detect_batch():
    response = client.post("/detect/batch", json={
        "transactions": [
            {"txn_id": "TXN-B001", "amount": 450.00,
             "txn_type": "ach_debit", "hour": 10},
            {"txn_id": "TXN-B002", "amount": 48000.00,
             "txn_type": "wire_transfer", "hour": 23},
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["total"]   == 2
    assert data["flagged"] >= 1

def test_stats_endpoint():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_analyzed"  in data
    assert "detection_rate"  in data
    assert "total_flagged"   in data

def test_invalid_transaction():
    response = client.post("/detect", json={
        "txn_id":   "TXN-BAD",
        "amount":   -500,
        "txn_type": "wire_transfer",
        "hour":     10
    })
    assert response.status_code == 422
