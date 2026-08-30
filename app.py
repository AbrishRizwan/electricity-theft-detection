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
    page_title="Electricity Theft Detection System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
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

    # Linear interpolation & forward/backward fill
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
    st.error("Error: Model artifact 'xgboost_optimized_detector.joblib' not found in root directory.")
    st.stop()

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("⚡ Control Panel")
threshold = st.sidebar.slider(
    "Audit Sensitivity Threshold",
    min_value=0.30,
    max_value=0.75,
    value=0.50,
    step=0.05,
    help="Higher threshold = targeted high-certainty audits. Lower threshold = aggressive theft sweep."
)

uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"], key="sidebar_uploader")

# ---------------------------------------------------------
# Application Header & Landing State
# ---------------------------------------------------------
st.title("⚡ Smart Grid Electricity Theft Detection System")
st.markdown("Automated non-technical loss (theft) detection using tree-based gradient boosting.")

raw_df = None

# If no file uploaded via sidebar, show prominent main-screen upload / demo loader
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
else:
    st.info("👋 **Welcome!** Upload a smart meter CSV file below, or load the built-in sample dataset to run the model.")
    col_upload, col_demo = st.columns([2, 1])
    
    with col_upload:
        main_uploaded_file = st.file_uploader("Choose a smart meter CSV file", type=["csv"], key="main_uploader")
        if main_uploaded_file is not None:
            raw_df = pd.read_csv(main_uploaded_file)
            
    with col_demo:
        st.write("**Or test with bundled sample:**")
        if st.button("🚀 Load 100-Meter Demo Data", use_container_width=True):
            if os.path.exists("sample_smart_meter_data.csv"):
                raw_df = pd.read_csv("sample_smart_meter_data.csv")
                st.session_state["demo_loaded"] = True
            else:
                st.warning("Sample CSV file not found on server.")

if raw_df is None and st.session_state.get("demo_loaded", False):
    if os.path.exists("sample_smart_meter_data.csv"):
        raw_df = pd.read_csv("sample_smart_meter_data.csv")

# ---------------------------------------------------------
# Inference & Dashboard Display
# ---------------------------------------------------------
if raw_df is not None:
    with st.spinner("Extracting time-series features and scoring meters..."):
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

    # KPI Metrics
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    total_meters = len(results_df)
    flagged_theft = int(theft_predictions.sum())
    flagged_pct = (flagged_theft / total_meters) * 100
    avg_risk = theft_probabilities.mean() * 100

    col1.metric("Total Meters Analyzed", f"{total_meters:,}")
    col2.metric("Flagged for On-Site Audit", f"{flagged_theft:,}", delta=f"{flagged_pct:.1f}% flag rate", delta_color="inverse")
    col3.metric("Normal Meters", f"{total_meters - flagged_theft:,}")
    col4.metric("Average Network Risk", f"{avg_risk:.1f}%")

    # Actionable Audit Queue Table
    st.subheader("🔍 Prioritized Inspection Queue")
    flagged_meters = results_df[results_df["Predicted_Status"] == "Theft / Suspicious"].sort_values(
        by="Theft_Probability", ascending=False
    )
    
    if len(flagged_meters) > 0:
        st.dataframe(flagged_meters, use_container_width=True, height=260)
        csv_data = flagged_meters.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Priority Audit List (CSV)",
            data=csv_data,
            file_name="electricity_theft_audit_report.csv",
            mime="text/csv"
        )
    else:
        st.success(f"No meters exceeded the current sensitivity threshold ({threshold:.2f}).")

    # Consumer Time-Series Profile Inspector
    st.divider()
    st.subheader("📈 Consumer Consumption Profile Inspector")
    
    # Filter selection: Flagged vs All
    filter_mode = st.radio(
        "Dropdown Selection Mode:",
        options=["Flagged Suspects Only (Ranked by Risk)", "All Analyzed Meters"],
        horizontal=True
    )
    
    if filter_mode.startswith("Flagged") and len(flagged_meters) > 0:
        available_options = flagged_meters["CONS_NO"].tolist()
    else:
        available_options = results_df.sort_values(by="Theft_Probability", ascending=False)["CONS_NO"].tolist()

    selected_consumer = st.selectbox(
        "Select a Consumer ID to inspect daily load profile:",
        options=available_options
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
            name='Daily Consumption (kWh)',
            line=dict(color='#d62728' if consumer_prob >= threshold else '#1f77b4', width=1.5)
        ))

        status_tag = "SUSPICIOUS / THEFT" if consumer_prob >= threshold else "NORMAL CONSUMER"
        status_color = "red" if consumer_prob >= threshold else "green"

        fig.update_layout(
            title=f"Consumption Timeline: <b>{selected_consumer}</b> — Status: <span style='color:{status_color}'>{status_tag} (Risk: {consumer_prob:.2%})</span>",
            xaxis_title="Date",
            yaxis_title="Daily Usage (kWh)",
            template="plotly_white",
            hovermode="x unified",
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)
