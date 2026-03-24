import os
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from models.anomaly_detector import AnomalyDetector
from data.generate_data import generate_transactions, save_to_sqlite

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Anomaly Detector",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Financial Anomaly Detection Dashboard")
st.caption("Detects suspicious transactions using Z-score, IQR, and Isolation Forest")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data  # cache so the DB isn't re-read on every interaction
def load_data():
    db_path = 'data/transactions.db'
    if not os.path.exists(db_path):
        os.makedirs('data', exist_ok=True)
        save_to_sqlite(generate_transactions(), db_path)
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql('SELECT * FROM transactions', conn)
    conn.close()
    return df

df_raw = load_data()

# ── Sidebar controls ───────────────────────────────────────────────────────────
st.sidebar.header("Detection settings")

zscore_threshold = st.sidebar.slider(
    "Z-score threshold", min_value=1.5, max_value=5.0,
    value=3.0, step=0.1,
    help="Higher = fewer flags. Lower = more sensitive."
)
iqr_multiplier = st.sidebar.slider(
    "IQR multiplier", min_value=1.0, max_value=3.0,
    value=1.5, step=0.1
)
txn_types = st.sidebar.multiselect(
    "Transaction types",
    options=df_raw['txn_type'].unique().tolist(),
    default=df_raw['txn_type'].unique().tolist()
)

df_filtered = df_raw[df_raw['txn_type'].isin(txn_types)]

# ── Run detection ──────────────────────────────────────────────────────────────
detector = AnomalyDetector()
df = detector.run_all(df_filtered, zscore_threshold, iqr_multiplier)

anomalies = df[df['predicted_anomaly'] == True]

# ── Metric cards ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total transactions",  f"{len(df):,}")
col2.metric("Anomalies detected",  f"{len(anomalies):,}",
            delta=f"{len(anomalies)/len(df)*100:.1f}% rate",
            delta_color="inverse")
col3.metric("Flagged value",
            f"${anomalies['amount'].sum():,.0f}")
col4.metric("High-risk transactions",
            f"{len(anomalies[anomalies['risk_level']=='High']):,}")

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.6, 1])

with col_left:
    st.subheader("Transaction amounts — normal vs anomalous")
    fig = px.scatter(
        df, x='timestamp', y='amount',
        color='predicted_anomaly',
        color_discrete_map={False: '#378ADD', True: '#E24B4A'},
        labels={'predicted_anomaly': 'Anomaly', 'amount': 'Amount ($)'},
        hover_data=['txn_id', 'txn_type', 'z_score', 'risk_level'],
        opacity=0.7,
    )
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Risk level distribution")
    risk_counts = anomalies['risk_level'].value_counts()
    fig2 = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color=risk_counts.index,
        color_discrete_map={
            'High': '#E24B4A', 'Medium': '#EF9F27', 'Low': '#639922'
        },
        hole=0.45,
    )
    fig2.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ── Flagged transactions table ──────────────────────────────────────────────────
st.subheader(f"Flagged transactions ({len(anomalies)})")

display_cols = ['txn_id', 'timestamp', 'txn_type', 'amount',
                'z_score', 'iso_score', 'anomaly_score', 'risk_level']

st.dataframe(
    anomalies[display_cols]
        .sort_values('anomaly_score', ascending=False)
        .reset_index(drop=True),
    use_container_width=True,
    column_config={
        "amount":        st.column_config.NumberColumn("Amount ($)",   format="$%.2f"),
        "anomaly_score": st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0, max_value=1),
        "risk_level":    st.column_config.TextColumn("Risk"),
    }
)