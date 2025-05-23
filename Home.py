import os, mlflow

# Set up MLflow tracking
MLRUNS_DIR = "/tmp/mlruns"
os.makedirs(MLRUNS_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
mlflow.set_experiment("ai-text-monitoring")

import streamlit as st
import re
import joblib
import streamlit.components.v1 as components
import gzip
import time

# Load vectorizer and model
vectorizer = joblib.load("vectorizer.pkl")
with gzip.open("model_compressed.pkl.gz", "rb") as f:
    model = joblib.load(f)

# Preprocess input
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\n', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Page title
st.title("Human vs AI Text Classifier")
st.info("For the best result, please enter text containing **at least 350 words**.")

# Input
text = st.text_area("Paste your text here:")

if st.button("Classify"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        start_time = time.time()

        # Preprocess and transform input
        cleaned_text = preprocess_text(text)
        vect_text = vectorizer.transform([cleaned_text])

        # Make prediction and get confidence
        prediction = model.predict(vect_text)[0]
        probabilities = model.predict_proba(vect_text)[0]
        confidence = probabilities[prediction] * 100

        # Label mapping
        label = "AI-generated" if prediction == 1 else "Human-written"

        # Output results
        st.subheader("Classification:")
        st.write(f"**{label}** (Confidence: {confidence:.2f}%)")

        latency = (time.time() - start_time) * 1000

        # Log to MLflow
        with mlflow.start_run(run_name = "inference", nested = True) as run:
            mlflow.log_params({
                "text_length": len(text),
                "predicted_class": int(prediction)
            })
            mlflow.log_metrics({
                "confidence": confidence,
                "latency_ms": latency
            })

        st.session_state["_t0"] = time.time()
        
        st.divider()
        st.subheader("MLflow Tracking Logs for This Prediction")
    
        st.markdown("**Parameters**")
        st.json({
            "text_length": len(text),
            "predicted_class": int(prediction)
        })
    
        st.markdown("**Metrics**")
        st.json({
            "latency_ms": round(latency,5),
            "ai_probability": confidence
        })
    
        st.markdown(f"**Run ID:** `{run.info.run_id}`")
