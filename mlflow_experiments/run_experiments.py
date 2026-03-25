import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
import mlflow
import mlflow.sklearn
from itertools import product
from models.anomaly_detector import AnomalyDetector


def load_data():
    conn = sqlite3.connect('data/transactions.db')
    df = pd.read_sql('SELECT * FROM transactions', conn)
    conn.close()
    return df


def run_experiment():
    """
    Run multiple experiment configurations and log every result to MLflow.
    This is exactly what a data science team does before promoting a model
    to production — systematic parameter search with full auditability.
    """

    # Parameter grid — every combination will become one logged run
    param_grid = {
        'contamination':   [0.03, 0.04, 0.05, 0.06, 0.08],
        'z_threshold':     [2.5, 3.0, 3.5],
        'iqr_multiplier':  [1.2, 1.5, 1.8],
    }

    df = load_data()

    # Name your experiment — shows up as the header in MLflow UI
    mlflow.set_experiment("anomaly-detection-experiment")

    best_run_id    = None
    best_detection = 0

    # Generate all combinations
    combos = list(product(
        param_grid['contamination'],
        param_grid['z_threshold'],
        param_grid['iqr_multiplier'],
    ))

    print(f"Running {len(combos)} experiment combinations...")

    for contamination, z_threshold, iqr_multiplier in combos:

        with mlflow.start_run():

            # ── Log parameters ──────────────────────────────────────────
            mlflow.log_param("contamination",  contamination)
            mlflow.log_param("z_threshold",    z_threshold)
            mlflow.log_param("iqr_multiplier", iqr_multiplier)
            mlflow.log_param("n_estimators",   100)
            mlflow.log_param("algorithm",      "IsolationForest")

            # ── Run detection ────────────────────────────────────────────
            detector = AnomalyDetector(contamination=contamination)
            results  = detector.run_all(
                df,
                zscore_threshold=z_threshold,
                iqr_multiplier=iqr_multiplier,
            )

            flagged = results[results['predicted_anomaly'] == True]

            # ── Calculate metrics ────────────────────────────────────────
            total             = len(results)
            n_anomalies       = len(flagged)
            detection_rate    = round(n_anomalies / total * 100, 2)
            high_risk_count   = len(flagged[flagged['risk_level'] == 'High'])
            medium_risk_count = len(flagged[flagged['risk_level'] == 'Medium'])
            avg_z_score       = round(flagged['z_score'].mean(), 3) if n_anomalies > 0 else 0
            flagged_value     = round(flagged['amount'].sum(), 2)

            # ── Log metrics ──────────────────────────────────────────────
            mlflow.log_metric("anomalies_detected",  n_anomalies)
            mlflow.log_metric("detection_rate_pct",  detection_rate)
            mlflow.log_metric("high_risk_count",     high_risk_count)
            mlflow.log_metric("medium_risk_count",   medium_risk_count)
            mlflow.log_metric("avg_z_score_flagged", avg_z_score)
            mlflow.log_metric("total_flagged_value", flagged_value)

            # ── Log the model itself ─────────────────────────────────────
            mlflow.sklearn.log_model(
                detector.iso_forest,
                name="isolation_forest_model",
            )

            # ── Tag the best run ─────────────────────────────────────────
            run_id = mlflow.active_run().info.run_id
            if n_anomalies > best_detection:
                best_detection = n_anomalies
                best_run_id    = run_id
                mlflow.set_tag("best_run", "true")

            print(
                f"contamination={contamination:.2f} | "
                f"z={z_threshold} | iqr={iqr_multiplier} | "
                f"flagged={n_anomalies} ({detection_rate}%)"
            )

    print(f"\nBest run ID: {best_run_id}")
    print(f"Best anomalies detected: {best_detection}")
    print("\nOpen MLflow UI: mlflow ui")
    return best_run_id


if __name__ == '__main__':
    run_experiment()