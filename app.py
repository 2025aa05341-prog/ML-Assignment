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

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Machine Learning Assignment-2")
st.caption("Breast Cancer Classification - BITS Pilani WILP")

# ------------------------
# Step 1: Dataset Loader [cite: 27-30]
# ------------------------
@st.cache_resource
def load_dataset():
    # Breast Cancer dataset meets the requirement of >12 features and >500 instances [cite: 30]
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return X, y, list(ds.feature_names), list(ds.target_names)

X, y, feature_names, target_names = load_dataset()
# Splitting data for evaluation [cite: 40]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ------------------------
# Sidebar: Model Selection [cite: 34-39, 92]
# ------------------------
MODEL_DIR = Path("model")
# All 6 required models [cite: 34-39]
MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
    "Decision Tree": MODEL_DIR / "decision_tree.pkl",
    "kNN": MODEL_DIR / "knn.pkl",
    "Naive Bayes": MODEL_DIR / "naive_bayes.pkl",
    "Random Forest": MODEL_DIR / "random_forest.pkl",
    "XGBoost": MODEL_DIR / "xgboost.pkl",
}

st.sidebar.header("Navigation")
existing = {name: p for name, p in MODEL_FILES.items() if p.exists()}

if not existing:
    st.error("No model artifacts found in /model. Please upload your .pkl files to GitHub.")
    st.stop()

selected_model_name = st.sidebar.selectbox("Select Model", list(existing.keys()))

# ------------------------
# Model Loading Utility
# ------------------------
@st.cache_resource
def load_trained_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

model = load_trained_model(existing[selected_model_name])

# ------------------------
# Step 6c: Display Evaluation Metrics [cite: 40-46, 93]
# ------------------------
y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

# Required Metrics: Acc, AUC, Prec, Rec, F1, MCC [cite: 41-46]
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_score) if y_score is not None else 0.0,
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "MCC Score": matthews_corrcoef(y_test, y_pred),
}

st.subheader(f"Metrics for {selected_model_name}")
cols = st.columns(6)
for i, (m_name, m_val) in enumerate(metrics.items()):
    cols[i].metric(m_name, f"{m_val:.4f}")

# ------------------------
# Step 6d: Confusion Matrix & Report [cite: 94]
# ------------------------
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm, index=target_names, columns=target_names))

with c2:
    st.write("### Classification Report")
    st.code(classification_report(y_test, y_pred, target_names=target_names))

# ------------------------
# Step 6a: Test Data Upload [cite: 91]
# ------------------------
st.divider()
st.subheader("Predict on Custom Test Data")
uploaded_file = st.file_uploader("Upload CSV (must have 30 feature columns)", type="csv")

if uploaded_file:
    try:
        test_df = pd.read_csv(uploaded_file)
        # Reindexing to ensure the column order matches the training features
        test_df = test_df.reindex(columns=feature_names, fill_value=0)
        
        preds = model.predict(test_df)
        
        # Corrected format to avoid syntax error
        out = pd.DataFrame({"prediction": preds.astype(int)})
        
        st.write("### Prediction Results")
        st.dataframe(out)
        st.download_button("Download Predictions", out.to_csv(index=False), "predictions.csv")
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
