import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, roc_auc_score, 
    confusion_matrix, classification_report
)

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(page_title="ML Assignment 2 - Breast Cancer", layout="wide")
st.title("Machine Learning Assignment-2")
st.subheader("Breast Cancer Classification Dashboard")

# ------------------------
# Step 1: Dataset Loader [cite: 27]
# ------------------------
@st.cache_resource
def load_data():
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return X, y, ds.target_names

X, y, target_names = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Step 6b: Model Selection Dropdown [cite: 92]
# ------------------------
MODEL_DIR = Path("model")
MODEL_MAPPING = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox("Select ML Model", list(MODEL_MAPPING.keys()))

def load_pickled_model(name):
    path = MODEL_DIR / MODEL_MAPPING[name]
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

model = load_pickled_model(model_choice)

if model is None:
    st.error(f"Error: {MODEL_MAPPING[model_choice]} not found in /model folder. Run your training script first.")
    st.stop()

# ------------------------
# Step 6c: Display Evaluation Metrics [cite: 93]
# ------------------------
y_pred = model.predict(X_test)
# Required metrics: Acc, AUC, Prec, Rec, F1, MCC [cite: 40, 41, 42, 43, 44, 45, 46]
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred),
}

if hasattr(model, "predict_proba"):
    metrics["AUC Score"] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
else:
    metrics["AUC Score"] = 0.0

st.write(f"### Performance: {model_choice}")
cols = st.columns(6)
for i, (name, val) in enumerate(metrics.items()):
    cols[i].metric(name, round(val, 4))

# ------------------------
# Step 6d: Confusion Matrix & Report [cite: 94]
# ------------------------
col1, col2 = st.columns(2)
with col1:
    st.write("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm, index=target_names, columns=target_names))

with col2:
    st.write("#### Classification Report")
    st.code(classification_report(y_test, y_pred, target_names=target_names))

# ------------------------
# Step 6a: Test Data Upload [cite: 91]
# ------------------------
st.divider()
st.write("### Custom Prediction")
uploaded_file = st.file_uploader("Upload test CSV data for prediction", type="csv")

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    # Ensure columns match training data
    test_df = test_df.reindex(columns=X.columns, fill_value=0)
    preds = model.predict(test_df)
    
    # FIXED: Corrected syntax here by removing extra parenthesis
    out = pd.DataFrame({"prediction": preds.astype(int)})
    
    st.write("#### Results")
    st.dataframe(out)
    st.download_button("Download Predictions", out.to_csv(index=False), "predictions.csv")
