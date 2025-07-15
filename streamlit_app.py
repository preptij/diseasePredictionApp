import streamlit as st
import pandas as pd
import joblib
import os
import re
import spacy
from spacy.matcher import PhraseMatcher
import markdown2
from io import BytesIO

from diseasePredictionApp.notebooks.training import label_encoder
from nlp_project.disease_symptom_prediction.streamlit_app import prediction, symptoms, precautions

# ========== File Paths ==========
MODEL_PATH = "models/disease_model.joblib"
VECTORIZER_PATH = "models/vectorizer.joblib"
DESC_PATH = "data/symptom_description.csv"
PRECAUTION_PATH = "data/symptom_precaution.csv"

# ========== Validate File Existence ==========
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error("⚠️ Model or vectorizer file not found. Please train and save them first.")
    st.stop()
def generate_markdown_report(input_text, prediction, symptoms, precautions):
    md = f"# Patient Symptom Summary\n\n"
    md += f"**Input:** {input_text}\n\n"
    md += f"**Predicted Disease:** {prediction}\n\n"
    md += f"**Extracted Symptoms:**\n"
    for s in symptoms:
        md += f"- {s}\n"
    md += f"\n**Recommended Precautions:**\n"
    for p in precautions:
        md += f"- {p}\n"
    return md

# ========== Load Model and Vectorizer ==========
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ========== Utility Functions ==========
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load Precaution Data
precautions_df = pd.read_csv(PRECAUTION_PATH)
precautions_map = precautions_df.set_index('Disease').T.to_dict('list')

def get_precautions(disease):
    return precautions_map.get(disease, ["N/A"])

# Load Symptom Descriptions for NER
desc_df = pd.read_csv(DESC_PATH)
desc_df['clean_text'] = desc_df['Description'].apply(clean_text)

# Initialize spaCy and PhraseMatcher
nlp = spacy.load("en_core_web_sm")
patterns = [nlp.make_doc(text) for text in desc_df['clean_text'].unique()]
matcher = PhraseMatcher(nlp.vocab)
matcher.add("SYMPTOM", patterns)

def extract_entities(text):
    doc = nlp(clean_text(text))
    matches = matcher(doc)
    return [doc[start:end].text for match_id, start, end in matches]

# ========== Streamlit UI ==========
st.set_page_config(page_title="Symptom-Based Disease Predictor", layout="centered")
st.title("🩺 Symptom-Based Disease Predictor")

user_input = st.text_area("Describe your symptoms:", "fatigue, vomiting, chest pain")

from weasyprint import HTML

def generate_pdf_report(markdown_report):
    html_report = markdown2.markdown(markdown_report)
    pdf_bytes = HTML(string=html_report).write_pdf()
    return pdf_bytes

if st.button("Predict Disease"):
    ...
    markdown_report = generate_markdown_report(user_input, prediction, symptoms, precautions)
    pdf_bytes = generate_pdf_report(markdown_report)

    st.download_button(
        label="📄 Download Report as PDF",
        data=pdf_bytes,
        file_name="symptom_summary.pdf",
        mime="application/pdf"
    )

    st.success(f"🧠 Predicted Disease: **{prediction}**")

    st.markdown("### 🔍 Extracted Symptoms:")
    if symptoms:
        for s in symptoms:
            st.markdown(f"- {s}")
    else:
        st.markdown("*No symptom matches found*")

    st.markdown("### 🛡️ Recommended Precautions:")
    for p in precautions:
        st.markdown(f"- {p}")

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizerFast.from_pretrained("notebooks/outputs/transformer_model")
model = DistilBertForSequenceClassification.from_pretrained("notebooks/outputs/transformer_model")
label_list = label_encoder.classes_  # Save/load from disk too

def predict_with_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted = torch.argmax(outputs.logits, dim=1).item()
    return label_list[predicted]