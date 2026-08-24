import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Electricity Theft Detection System",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------------------------------------
# Helper Function: Feature Engineering Pipeline
# ---------------------------------------------------------
@st.cache_data
def process_and_extract_features(raw_df):
    """
    Cleans raw smart meter time-series data and computes
    the 13 statistical and behavioral features for XGBoost.
    """
    df = raw_df.copy()
    
    # Save consumer IDs if column exists
    if "CONS_NO" in df.columns:
        consumer_ids = df["CONS_NO"]
        ts_data = df.drop(columns=["CONS_NO", "FLAG"], errors="ignore")
    else:
        consumer_ids = pd.Series([f"METER_{i}" for i in range(len(df))])
        ts_data = df.copy()

    # Step 1: Clean missing values via vector interpolation
    ts_cleaned = (
        ts_data.T.interpolate(method="linear")
        .ffill()
        .bfill()
        .fillna(0)
        .T.astype("float32")
    )

    # Step 2: Extract 13 Behavioral Features
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
    
    # Recent 30-day drop calculation
    recent_30_mean = ts_cleaned.iloc[:, -30:].mean(axis=1)
    features["drop_ratio"] = recent_30_mean / (features["mean"] + eps)

    # Clean any accidental inf/nan values
    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    
    return consumer_ids, ts_cleaned, features

# ---------------------------------------------------------
# Load Model Artifact
# ---------------------------------------------------------
@st.cache_resource
def load_detector_model():
    return joblib.load("xgboost_optimized_detector.joblib")

try:
    model = load_detector_model()
except Exception:
    st.error("Error: 'xgboost_optimized_detector.joblib' not found. Please ensure the model file is in the same directory.")
    st.stop()

# ---------------------------------------------------------
# Application Header & Sidebar
# ---------------------------------------------------------
st.title("⚡ Smart Grid Electricity Theft Detection System")
st.markdown("Automated non-technical loss (theft) detection using tree-based gradient boosting.")

st.sidebar.header("🔧 Inspection Controls")
threshold = st.sidebar.slider(
    "Audit Sensitivity Threshold",
    min_value=0.30,
    max_value=0.70,
    value=0.50,
    step=0.05,
    help="Lower threshold catches more theft (higher recall); higher threshold reduces false alarms (higher precision)."
)

uploaded_file = st.sidebar.file_uploader("Upload Smart Meter CSV", type=["csv"])

# ---------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded: {raw_df.shape[0]} Consumers")
    
    with st.spinner("Extracting time-series features and scoring meters..."):
        consumer_ids, ts_cleaned, features = process_and_extract_features(raw_df)
        
        # Predict continuous theft probabilities
        theft_probabilities = model.predict_proba(features)[:, 1]
        theft_predictions = (theft_probabilities >= threshold).astype(int)

    # Build results table
    results_df = pd.DataFrame({
        "CONS_NO": consumer_ids,
        "Theft_Probability": np.round(theft_probabilities, 4),
        "Predicted_Status": np.where(theft_predictions == 1, "Theft / Suspicious", "Normal"),
        "Mean_Daily_kWh": np.round(features["mean"], 2),
        "30Day_Drop_Ratio": np.round(features["drop_ratio"], 2),
        "Zero_Days_Ratio": np.round(features["zero_days_ratio"], 2)
    })

    # Display KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total_meters = len(results_df)
    flagged_theft = int(theft_predictions.sum())
    flagged_pct = (flagged_theft / total_meters) * 100
    avg_risk = theft_probabilities.mean() * 100

    col1.metric("Total Meters Analyzed", f"{total_meters:,}")
    col2.metric("Flagged for On-Site Audit", f"{flagged_theft:,}", delta=f"{flagged_pct:.1f}% flag rate", delta_color="inverse")
    col3.metric("Normal Meters", f"{total_meters - flagged_theft:,}")
    col4.metric("Average Network Risk", f"{avg_risk:.1f}%")

    # Flagged Accounts Inspection Section
    st.subheader("🔍 Prioritized Inspection Queue")
    flagged_meters = results_df[results_df["Predicted_Status"] == "Theft / Suspicious"].sort_values(
        by="Theft_Probability", ascending=False
    )
    
    st.dataframe(flagged_meters, use_container_width=True, height=250)

    # Download Audit Report Button
    csv_data = flagged_meters.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Priority Audit List (CSV)",
        data=csv_data,
        file_name="electricity_theft_audit_report.csv",
        mime="text/csv"
    )

    # Interactive Time-Series Consumer Analysis
    st.subheader("📈 Consumer Consumption Profile Inspector")
    selected_consumer = st.selectbox(
        "Select a Consumer ID to visualize daily load curve:",
        options=results_df["CONS_NO"].tolist()
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
            line=dict(color='#1f77b4', width=1.5)
        ))

        status_color = "red" if consumer_prob >= threshold else "green"
        status_text = "SUSPICIOUS / THEFT" if consumer_prob >= threshold else "NORMAL CONSUMER"

        fig.update_layout(
            title=f"Consumption Timeline for {selected_consumer} — Status: <span style='color:{status_color}'>{status_text} (Risk Score: {consumer_prob:.2%})</span>",
            xaxis_title="Date",
            yaxis_title="Electricity Consumption (kWh)",
            template="plotly_white",
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Please upload a smart meter CSV file in the sidebar to begin analysis.")