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
# Page Configuration [cite: 90]
# ------------------------
st.set_page_config(page_title="ML Assignment 2 - Breast Cancer", layout="wide")
st.title("Machine Learning Assignment-2")
st.markdown("### Classification on Breast Cancer Dataset")

# ------------------------
# Constants & Model Mapping [cite: 34-39, 55]
# ------------------------
MODEL_DIR = Path("model")
MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
    "Decision Tree": MODEL_DIR / "decision_tree.pkl",
    "kNN": MODEL_DIR / "knn.pkl",
    "Naive Bayes": MODEL_DIR / "naive_bayes.pkl",
    "Random Forest": MODEL_DIR / "random_forest.pkl",
    "XGBoost": MODEL_DIR / "xgboost.pkl",
}

# ------------------------
# Dataset Loader [cite: 27-30]
# ------------------------
@st.cache_resource
def load_dataset():
    # Breast Cancer: 30 features, 569 instances (meets >12 features, >500 instances) [cite: 30]
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return X, y, list(ds.feature_names), list(ds.target_names)

X, y, feature_names, target_names = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------
# Sidebar: Model Selection [cite: 92]
# ------------------------
st.sidebar.header("Controls")
existing_models = [name for name, path in MODEL_FILES.items() if path.exists()]

if not existing_models:
    st.error("No .pkl files found in /model directory. Please upload trained models.")
    st.stop()

selected_model = st.sidebar.selectbox("Select Model", existing_models)

@st.cache_resource
def get_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

model = get_model(MODEL_FILES[selected_model])

# ------------------------
# Evaluation & Metrics Display [cite: 40-46, 93, 94]
# ------------------------
y_pred = model.predict(X_test)
y_score = None
if hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X_test)[:, 1]

# Mandatory Metrics [cite: 41-46]
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC Score": roc_auc_score(y_test, y_score) if y_score is not None else 0.0,
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "MCC Score": matthews_corrcoef(y_test, y_pred),
}

st.subheader(f"Performance Metrics: {selected_model}")
cols = st.columns(6)
for i, (m_name, m_val) in enumerate(metrics.items()):
    cols[i].metric(m_name, f"{m_val:.4f}")

st.divider()
col_cm, col_cr = st.columns(2)

with col_cm:
    st.write("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm, index=target_names, columns=target_names), use_container_width=True)

with col_cr:
    st.write("#### Classification Report")
    st.code(classification_report(y_test, y_pred, target_names=target_names))

# ------------------------
# Dataset Upload (Step 6a) [cite: 91]
# ------------------------
st.divider()
st.subheader("Predict on New Data")
st.info("Upload a CSV file containing test data (30 features required).")
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    try:
        test_df = pd.read_csv(uploaded_file)
        # Ensure correct column order
        test_df = test_df.reindex(columns=feature_names, fill_value=0)
        
        preds = model.predict(test_df)
        
        # Corrected format [fixes the SyntaxError]
        out = pd.DataFrame({"prediction": preds.astype(int)})
        
        st.write("#### Prediction Results")
        st.dataframe(out, use_container_width=True)
        st.download_button("Download Predictions", out.to_csv(index=False), "predictions.csv")
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
