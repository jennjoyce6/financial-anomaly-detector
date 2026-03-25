import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """
    Three-algorithm anomaly detection suite for financial transactions.

    Algorithms used:
    - Z-score:        Fast, interpretable. Good for normally distributed amounts.
    - IQR method:     Robust to skewed distributions. No assumptions about shape.
    - Isolation Forest: ML-based. Catches multi-dimensional anomalies (amount + hour).
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(
            contamination=0.06,  # expect ~3% anomaly rate
            random_state=42,
            n_estimators=100,
        )

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw transaction data."""
        df = df.copy()
        df['hour']        = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_offhours'] = ((df['hour'] < 7) | (df['hour'] > 20)).astype(int)
        df['amount_log']  = np.log1p(df['amount'])  # log-transform skewed amounts
        return df

    def zscore_detection(self, df: pd.DataFrame,
                         threshold: float = 3.0) -> pd.Series:
        """
        Flag transactions where amount z-score exceeds threshold.
        Z-score = (value - mean) / std_deviation
        """
        mean   = df['amount'].mean()
        std    = df['amount'].std()
        zscores = (df['amount'] - mean) / std
        return zscores, zscores.abs() > threshold

    def iqr_detection(self, df: pd.DataFrame,
                      multiplier: float = 1.5) -> pd.Series:
        """
        Flag transactions outside Q1 - 1.5*IQR and Q3 + 1.5*IQR.
        More robust than Z-score for skewed financial data.
        """
        Q1  = df['amount'].quantile(0.25)
        Q3  = df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        return (df['amount'] < lower) | (df['amount'] > upper)

    def isolation_forest_detection(self, df: pd.DataFrame) -> pd.Series:
        """
        ML-based detection using Isolation Forest.
        Catches anomalies across multiple dimensions simultaneously
        (e.g., a medium-sized amount at 3am is suspicious even if the
        amount alone looks normal).
        """
        features = df[['amount_log', 'hour', 'is_weekend', 'is_offhours']]
        scaled   = self.scaler.fit_transform(features)
        preds    = self.iso_forest.fit_predict(scaled)
        # IsolationForest returns -1 for anomalies, 1 for normal
        scores   = self.iso_forest.score_samples(scaled)
        return scores, preds == -1

    def run_all(self, df: pd.DataFrame,
                zscore_threshold: float = 3.0,
                iqr_multiplier: float = 1.5) -> pd.DataFrame:
        """
        Run all three detectors and combine results.
        Returns enriched DataFrame with anomaly flags and scores.
        """
        df = self.add_features(df)

        zscores, z_flags       = self.zscore_detection(df, zscore_threshold)
        iqr_flags              = self.iqr_detection(df, iqr_multiplier)
        iso_scores, iso_flags  = self.isolation_forest_detection(df)

        df['z_score']          = zscores.round(3)
        df['z_flag']           = z_flags
        df['iqr_flag']         = iqr_flags
        df['iso_score']        = iso_scores.round(4)
        df['iso_flag']         = iso_flags

        # Consensus: flagged by 2+ algorithms = high confidence anomaly
        flag_count             = z_flags.astype(int) + iqr_flags.astype(int) + iso_flags.astype(int)
        df['anomaly_score']    = (flag_count / 3).round(2)
        df['predicted_anomaly'] = flag_count >= 1

        # Risk tier
        df['risk_level'] = pd.cut(
            df['anomaly_score'],
            bins=[-0.1, 0.34, 0.67, 1.01],
            labels=['Low', 'Medium', 'High']
        )

        return df