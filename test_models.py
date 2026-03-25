import sqlite3
import pandas as pd
from models.anomaly_detector import AnomalyDetector

# Load data
conn = sqlite3.connect('data/transactions.db')
df = pd.read_sql('SELECT * FROM transactions', conn)
conn.close()

# Run detection
detector = AnomalyDetector()
results = detector.run_all(df)

# Summary
flagged = results[results['predicted_anomaly'] == True]
print(f'Total transactions:    {len(results):,}')
print(f'Anomalies detected:    {len(flagged):,}')
print(f'Detection rate:        {len(flagged)/len(results)*100:.1f}%')

print()
print('Risk breakdown:')
print(flagged['risk_level'].value_counts())

print()
print('Sample flagged transactions:')
print(
    flagged[['txn_id', 'amount', 'z_score', 'anomaly_score', 'risk_level']]
    .sort_values('anomaly_score', ascending=False)
    .head(5)
    .to_string()
)