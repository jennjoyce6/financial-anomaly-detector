import pandas as pd
import numpy as np
from faker import Faker
import sqlite3
from datetime import datetime, timedelta
import random

fake = Faker()
np.random.seed(42)

def generate_transactions(n_normal=5000, n_anomalies=150):
    """
    Generate synthetic financial transaction data with injected anomalies.
    Normal transactions cluster around typical business patterns.
    Anomalies are outliers in amount, frequency, or time-of-day.
    """
    transactions = []

    # --- Normal transactions ---
    for _ in range(n_normal):
        txn_type = random.choice(
            ['wire_transfer', 'ach_debit', 'vendor_payment',
             'payroll', 'intl_payment']
        )
        # Normal amounts vary by type
        amount_ranges = {
            'wire_transfer':   (500,   15000),
            'ach_debit':       (50,    5000),
            'vendor_payment':  (200,   8000),
            'payroll':         (1000,  12000),
            'intl_payment':    (300,   10000),
        }
        low, high = amount_ranges[txn_type]
        amount = round(np.random.lognormal(
            mean=np.log((low + high) / 2), sigma=0.4
        ), 2)
        amount = max(low, min(high, amount))  # clip to range

        # Normal hours: business hours 8am-6pm
        hour = int(np.random.normal(13, 2.5))
        hour = max(8, min(18, hour))

        transactions.append({
            'txn_id':     f"TXN-{fake.unique.random_int(10000, 99999)}",
            'timestamp':  fake.date_time_between(
                              start_date='-90d', end_date='now'
                          ).replace(hour=hour),
            'amount':     amount,
            'txn_type':   txn_type,
            'sender':     fake.company(),
            'receiver':   fake.company(),
            'is_anomaly': 0,
        })

    # --- Injected anomalies ---
    anomaly_patterns = [
        # Pattern 1: unusually large amounts
        lambda: {
            'amount':     round(random.uniform(40000, 120000), 2),
            'hour':       random.randint(8, 18),
            'txn_type':   'wire_transfer',
        },
        # Pattern 2: off-hours transactions (midnight activity)
        lambda: {
            'amount':     round(random.uniform(500, 8000), 2),
            'hour':       random.randint(0, 5),
            'txn_type':   random.choice(['ach_debit', 'intl_payment']),
        },
        # Pattern 3: round-number structuring (just under reporting thresholds)
        lambda: {
            'amount':     round(random.choice([9999, 9998, 9997, 4999]) +
                               random.uniform(0, 0.99), 2),
            'hour':       random.randint(9, 17),
            'txn_type':   'wire_transfer',
        },
    ]

    for _ in range(n_anomalies):
        pattern = random.choice(anomaly_patterns)()
        transactions.append({
            'txn_id':     f"TXN-{fake.unique.random_int(10000, 99999)}",
            'timestamp':  fake.date_time_between(
                              start_date='-90d', end_date='now'
                          ).replace(hour=pattern['hour']),
            'amount':     pattern['amount'],
            'txn_type':   pattern['txn_type'],
            'sender':     fake.company(),
            'receiver':   fake.company(),
            'is_anomaly': 1,
        })

    df = pd.DataFrame(transactions)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle
    return df


def save_to_sqlite(df, db_path='data/transactions.db'):
    """Persist transactions to SQLite — shows SQL skills on your resume."""
    conn = sqlite3.connect(db_path)
    df.to_sql('transactions', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Saved {len(df)} transactions to {db_path}")


if __name__ == '__main__':
    df = generate_transactions()
    save_to_sqlite(df)
    print(df.head())
    print(f"\nAnomalies injected: {df['is_anomaly'].sum()}")