import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Grid Electricity Theft Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar collapsed by default
)

# ---------------------------------------------------------
# Helper: Feature Extraction Pipeline
# ---------------------------------------------------------
@st.cache_data
def process_and_extract_features(raw_df):
    df = raw_df.copy()
    
    if "CONS_NO" in df.columns:
        consumer_ids = df["CONS_NO"]
        ts_data = df.drop(columns=["CONS_NO", "FLAG"], errors="ignore")
    else:
        consumer_ids = pd.Series([f"METER_{i+1:04d}" for i in range(len(df))])
        ts_data = df.copy()

    # Linear interpolation & fill
    ts_cleaned = (
        ts_data.T.interpolate(method="linear")
        .ffill()
        .bfill()
        .fillna(0)
        .T.astype("float32")
    )

    eps = 1e-6
    features = pd.DataFrame(index=ts_cleaned.index)
    features["mean"] = ts_cleaned.mean(axis=1)
    features["std"] = ts_cleaned.std(axis=1)
    features["var"] = ts_cleaned.var(axis=1)
    features["median"] = ts_cleaned.median(axis=1)
    features["min"] = ts_cleaned.min(axis=1)
    features["max"] = ts_cleaned.max(axis=1)
    features["cv"] = features["std"] / (features["mean"] + eps)
    features["par"] = features["max"] / (features["mean"] + eps)
    features["skew"] = ts_cleaned.skew(axis=1).fillna(0)
    features["kurtosis"] = ts_cleaned.kurt(axis=1).fillna(0)
    features["zero_days_count"] = (ts_cleaned == 0.0).sum(axis=1)
    features["zero_days_ratio"] = features["zero_days_count"] / ts_cleaned.shape[1]
    
    recent_30_mean = ts_cleaned.iloc[:, -30:].mean(axis=1)
    features["drop_ratio"] = recent_30_mean / (features["mean"] + eps)

    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    return consumer_ids, ts_cleaned, features

# ---------------------------------------------------------
# Model Loading
# ---------------------------------------------------------
@st.cache_resource
def load_detector_model():
    return joblib.load("xgboost_optimized_detector.joblib")

try:
    model = load_detector_model()
except Exception:
    st.error("⚠️ Error: Model file 'xgboost_optimized_detector.joblib' not found. Please verify the file is uploaded.")
    st.stop()

# ---------------------------------------------------------
# Header & Front Controls
# ---------------------------------------------------------
st.title("⚡ Smart Grid Electricity Theft Detection System")
st.markdown("Automated non-technical loss (theft) detection using tree-based gradient boosting.")

# Top Main-Screen Control Box
with st.container():
    st.markdown("### 🎛️ Control Panel")
    ctrl_col1, ctrl_col2 = st.columns([1.5, 2.5])
    
    with ctrl_col1:
        threshold = st.slider(
            "🎯 Audit Sensitivity Threshold",
            min_value=0.30,
            max_value=0.80,
            value=0.50,
            step=0.05,
            help="Higher threshold = targeted high-certainty audits. Lower threshold = broad theft detection sweep."
        )
    
    with ctrl_col2:
        st.write("**Dataset Input:**")
        input_opt = st.radio(
            "Select Data Source:",
            ["🚀 Use Built-in 100-Meter Demo", "📁 Upload Custom CSV File"],
            horizontal=True
        )

# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
raw_df = None

if input_opt == "🚀 Use Built-in 100-Meter Demo":
    if os.path.exists("sample_smart_meter_data.csv"):
        raw_df = pd.read_csv("sample_smart_meter_data.csv")
    else:
        st.warning("Sample CSV file 'sample_smart_meter_data.csv' not found in root directory.")
else:
    uploaded_file = st.file_uploader("Upload Smart Meter CSV Data", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)

# ---------------------------------------------------------
# Inference & Dashboard Display
# ---------------------------------------------------------
if raw_df is not None:
    with st.spinner("Processing meter consumption data and calculating risk scores..."):
        consumer_ids, ts_cleaned, features = process_and_extract_features(raw_df)
        theft_probabilities = model.predict_proba(features)[:, 1]
        theft_predictions = (theft_probabilities >= threshold).astype(int)

    results_df = pd.DataFrame({
        "CONS_NO": consumer_ids,
        "Theft_Probability": np.round(theft_probabilities, 4),
        "Predicted_Status": np.where(theft_predictions == 1, "Theft / Suspicious", "Normal"),
        "Mean_Daily_kWh": np.round(features["mean"], 2),
        "30Day_Drop_Ratio": np.round(features["drop_ratio"], 2),
        "Zero_Days_Ratio": np.round(features["zero_days_ratio"], 2)
    })

    # KPI Summary Cards
    st.markdown("---")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_meters = len(results_df)
    flagged_theft = int(theft_predictions.sum())
    flagged_pct = (flagged_theft / total_meters) * 100
    avg_risk = theft_probabilities.mean() * 100

    kpi1.metric("Total Meters Analyzed", f"{total_meters:,}")
    kpi2.metric("Flagged for Audit", f"{flagged_theft:,}", delta=f"{flagged_pct:.1f}% Flag Rate", delta_color="inverse")
    kpi3.metric("Normal Meters", f"{total_meters - flagged_theft:,}")
    kpi4.metric("Average Risk Score", f"{avg_risk:.1f}%",
            help="Mean model score, not a calibrated probability. The real theft rate in the training data is about 8.5%.")
    st.caption("The model is tuned to catch theft (high recall), so it flags more meters than the true theft rate (~8.5%). Flagged meters need inspection; they are not confirmed theft.")

    # Prioritized Inspection Queue
    st.markdown("### 🔍 Prioritized Inspection Queue")
    flagged_meters = results_df[results_df["Predicted_Status"] == "Theft / Suspicious"].sort_values(
        by="Theft_Probability", ascending=False
    )
    
    if len(flagged_meters) > 0:
        st.dataframe(flagged_meters, use_container_width=True, height=240)
        csv_data = flagged_meters.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Priority Audit List (CSV)",
            data=csv_data,
            file_name="priority_theft_audit_list.csv",
            mime="text/csv"
        )
    else:
        st.success(f"No meters flagged above the active sensitivity threshold ({threshold:.2f}).")

    # Consumer Consumption Timeline Inspector
    st.markdown("---")
    st.markdown("### 📈 Consumer Consumption Profile Inspector")
    
    view_col1, view_col2 = st.columns([1, 2])
    with view_col1:
        view_filter = st.radio(
            "Filter Dropdown Accounts:",
            ["Flagged Suspects Only (Ranked by Risk)", "All Analyzed Meters"],
            horizontal=True
        )

    if view_filter.startswith("Flagged") and len(flagged_meters) > 0:
        dropdown_options = flagged_meters["CONS_NO"].tolist()
    else:
        dropdown_options = results_df.sort_values(by="Theft_Probability", ascending=False)["CONS_NO"].tolist()

    selected_consumer = st.selectbox(
        "Select Consumer ID to inspect time-series load curve:",
        options=dropdown_options
    )

    if selected_consumer:
        selected_idx = results_df[results_df["CONS_NO"] == selected_consumer].index[0]
        consumer_row = ts_cleaned.iloc[selected_idx]
        consumer_prob = theft_probabilities[selected_idx]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=consumer_row.index,
            y=consumer_row.values,
            mode='lines',
            name='Daily Usage (kWh)',
            line=dict(color='#e74c3c' if consumer_prob >= threshold else '#2ecc71', width=1.5)
        ))

        status_tag = "SUSPICIOUS / THEFT" if consumer_prob >= threshold else "NORMAL CONSUMER"
        status_color = "#e74c3c" if consumer_prob >= threshold else "#2ecc71"

        fig.update_layout(
            title=f"Consumption Timeline: <b>{selected_consumer}</b> — <span style='color:{status_color}'>{status_tag} (Risk: {consumer_prob:.2%})</span>",
            xaxis_title="Billing Date",
            yaxis_title="Daily Consumption (kWh)",
            template="plotly_white",
            hovermode="x unified",
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)
