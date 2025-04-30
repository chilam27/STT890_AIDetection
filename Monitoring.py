import os, mlflow

MLRUNS_DIR = "/tmp/mlruns"
os.makedirs(MLRUNS_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
mlflow.set_experiment("ai-text-monitoring")

import streamlit as st
import pandas as pd

st.title("Model Monitoring")

# Fetch the last 500 inference runs
exp = mlflow.get_experiment_by_name("ai-text-monitoring")
df = mlflow.search_runs(
    experiment_ids = [exp.experiment_id],
    filter_string = "tags.mlflow.runName = 'inference'",
    order_by = ["start_time DESC"],
    max_results = 500,
)

# If no inference runs yet, inject a dummy example
if df.empty or not any(col.startswith("metrics.") for col in df.columns):
    st.warning("No inference runs found yet — showing an example placeholder row.")
    example = {
        "run_id": "example-run",
        "experiment_id": exp.experiment_id,
        "status": "FINISHED",
        "artifact_uri": "",
        "start_time": pd.Timestamp.now().isoformat(),
        "end_time": pd.Timestamp.now().isoformat(),
        "params.text_length": 0,
        "params.predicted_class": 0,
        "metrics.confidence": 0.0,
        "metrics.latency_ms": 0.0,
    }
    df = pd.DataFrame([example])
    
# Parse timestamps & drop tz
df["time"] = pd.to_datetime(df["start_time"]) # mlflow returns start_time as ISO string (with +00:00)
if df["time"].dt.tz is not None:
    df["time"] = df["time"].dt.tz_convert(None) # Drop the timezone so st.line_chart can handle it

# Sort & re‐index
df = df.sort_values("time")
df_ts = df.set_index("time")

# Clean up column names & types
df_ts = df_ts.rename(columns = {
    "params.text_length": "text_length",
    "params.predicted_class": "predicted_class",
    "metrics.confidence": "confidence",
    "metrics.latency_ms": "latency_ms",
})

# Ensure metrics are numeric
df_ts["confidence"] = pd.to_numeric(df_ts["confidence"], errors = "coerce")
df_ts["latency_ms"] = pd.to_numeric(df_ts["latency_ms"], errors = "coerce")
df_ts["text_length"] = pd.to_numeric(df_ts["text_length"], errors = "coerce")
df_ts["predicted_class"] = pd.to_numeric(df_ts["predicted_class"], errors = "coerce")

# KPI Cards (last 50 runs)
latest = df_ts.head(50)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Latency (ms)", f"{latest['latency_ms'].mean():.1f}")
c2.metric("Avg Confidence (%)", f"{latest['confidence'].mean():.1f}")
c3.metric("Total Inferences", len(df_ts))
c4.metric("% AI-Generated", f"{(df_ts['predicted_class'] == 1).mean()*100:.1f}%")

st.markdown("---")

# Time-Series Trends
st.subheader("Latency over Time")
st.line_chart(df_ts["latency_ms"])

st.subheader("Confidence over Time")
st.line_chart(df_ts["confidence"])

st.markdown("---")

# Class Distribution
st.subheader("Predicted Class Distribution")
dist = df_ts["predicted_class"].map({0: "Human", 1: "AI"}).value_counts()
st.bar_chart(dist)

st.markdown("---")

# Recent Inference Logs
st.subheader("Recent Inference Logs")
st.dataframe(
    df_ts[[
        "run_id",
        "text_length",
        "predicted_class",
        "confidence",
        "latency_ms",
    ]].head(50),
    use_container_width = True,
)
