# Financial Anomaly Detection Dashboard

A production-style financial transaction monitoring system that detects 
suspicious activity using three statistical and machine learning algorithms, 
visualized through an interactive real-time dashboard.

**[Live Demo](https://financial-anomaly-detector-kdwrqnaah73fgv5pq37f3t.streamlit.app/)** | **[GitHub](https://github.com/jennjoyce6/financial-anomaly-detector)**

---

## What it does

Ingests financial transaction data, runs three anomaly detection algorithms 
in parallel, and surfaces flagged transactions with risk scores through an 
interactive Streamlit dashboard — enabling analysts to tune detection 
sensitivity in real time.

---

## Tech stack

| Layer | Technology |
|---|---|
| Dashboard | Streamlit, Plotly |
| Detection models | Scikit-learn (Isolation Forest), NumPy |
| Data layer | SQLite, SQLAlchemy, Pandas |
| Testing | Pytest |
| Language | Python 3.11 |

---

## Detection algorithms

Three algorithms run in parallel on every transaction. A transaction is 
flagged when two or more algorithms agree (consensus scoring):

**1. Z-score detection**
Flags transactions where the amount deviates more than N standard deviations 
from the mean. Fast and interpretable — good for normally distributed data.

**2. IQR method**
Flags transactions outside Q1 - 1.5×IQR and Q3 + 1.5×IQR bounds. More 
robust than Z-score for skewed financial distributions.

**3. Isolation Forest (ML)**
Catches multi-dimensional anomalies — a medium-sized transaction at 3am is 
suspicious even if the amount alone looks normal. Uses amount, hour, 
weekend flag, and off-hours flag as features.

---

## Architecture
```
financial-anomaly-detector/
├── app.py                  ← Streamlit dashboard (entry point)
├── data/
│   └── generate_data.py    ← Synthetic transaction generator (SQLite)
├── models/
│   └── anomaly_detector.py ← Z-score, IQR, Isolation Forest
├── tests/
│   └── test_detector.py    ← Pytest unit tests
└── requirements.txt
```

---

## Running locally
```bash
# 1. Clone the repo
git clone https://github.com/jennjoyce6/financial-anomaly-detector
cd financial-anomaly-detector

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate transaction data
python data/generate_data.py

# 5. Run the dashboard
streamlit run app.py
```

---

## Running tests
```bash
pytest tests/ -v
```

---

## Key results

- **5,150 transactions** processed across 5 transaction types
- **374 anomalies detected** at 7.3% detection rate
- **Three risk tiers** — High, Medium, Low with confidence scoring
- **Adjustable thresholds** — Z-score and IQR sensitivity tunable in real time
- **Full unit test coverage** via pytest

---

## Anomaly patterns detected

The synthetic dataset includes three injected anomaly patterns that mirror 
real-world financial fraud signals:

- **Large amount outliers** — wire transfers significantly above normal range
- **Off-hours activity** — transactions between midnight and 5am
- **Structuring patterns** — amounts just below common reporting thresholds 
  ($9,999, $4,999) — a known financial fraud technique

---

## If I were to extend this

- Connect to a live transaction stream via Kafka or AWS Kinesis
- Add email/Slack alerting when High-risk transactions are detected
- Replace synthetic data with anonymized public datasets (e.g. PaySim)
- Add a model retraining pipeline with MLflow experiment tracking
- Build a REST API wrapper so other services can query the detector