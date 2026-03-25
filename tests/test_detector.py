import pytest
import pandas as pd
import numpy as np
from models.anomaly_detector import AnomalyDetector

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