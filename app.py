# app.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# --- Page config ---
st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Machine Learning Assignment-2: Breast Cancer Classification")
st.caption("Deployment of 6 ML models with performance comparison.")

# --- Constants & Paths ---
MODEL_DIR = Path("model")
# Required models [cite: 34-39]
MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
    "Decision Tree": MODEL_DIR / "decision_tree.pkl",
    "kNN": MODEL_DIR / "knn.pkl",
    "Naive Bayes": MODEL_DIR / "naive_bayes.pkl",
    "Random Forest": MODEL_DIR / "random_forest.pkl",
    "XGBoost": MODEL_DIR / "xgboost.pkl",
}

# --- Load Dataset [cite: 27] ---
@st.cache_resource
def load_dataset():
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return X, y, list(ds.feature_names), list(ds.target_names)

X, y, feature_names, target_names = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Sidebar: Model Selection [cite: 92] ---
st.sidebar.header("Controls")
existing_models = [name for name, path in MODEL_FILES.items() if path.exists()]

if not existing_models:
    st.error("No model files found in /model directory. Please upload your .pkl files.")
    st.stop()

model_name = st.sidebar.selectbox("Select Model", existing_models)

# --- Load Selected Model ---
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

current_model = load_model(MODEL_FILES[model_name])

# --- Evaluation [cite: 40-46, 93] ---
y_pred = current_model.predict(X_test)
y_proba = current_model.predict_proba(X_test)[:, 1] if hasattr(current_model, "predict_proba") else None

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0,
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred),
}

st.subheader(f"Evaluation Metrics for {model_name}")
cols = st.columns(6)
for i, (m_name, m_val) in enumerate(metrics.items()):
    cols[i].metric(m_name, f"{m_val:.4f}")

# --- Confusion Matrix & Report [cite: 94] ---
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm, index=target_names, columns=target_names))

with col2:
    st.write("### Classification Report")
    st.code(classification_report(y_test, y_pred, target_names=target_names))

# --- CSV Upload for Test Data [cite: 91] ---
st.divider()
st.subheader("Upload Custom Test Data")
uploaded_file = st.file_uploader("Upload CSV (ensure 30 feature columns match exactly)", type="csv")

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    # Validation
    if all(col in test_df.columns for col in feature_names):
        test_df = test_df[feature_names]
        preds = current_model.predict(test_df)
        
        # Corrected prediction DataFrame format
        out = pd.DataFrame({"prediction": preds.astype(int)})
        
        st.write("### Predictions")
        st.dataframe(out)
        st.download_button("Download Predictions", out.to_csv(index=False), "predictions.csv")
    else:
        st.error("CSV columns do not match Breast Cancer feature names.")
