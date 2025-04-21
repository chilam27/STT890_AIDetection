import streamlit as st
import re
import joblib
import streamlit.components.v1 as components
import gzip

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

# Input
text = st.text_area("Paste your text here:")

if st.button("Classify"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
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
