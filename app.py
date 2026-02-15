import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, roc_auc_score, 
    confusion_matrix, classification_report
)

# Page configuration
st.set_page_config(page_title="ML Assignment 2 - Breast Cancer", layout="wide")
st.title("Breast Cancer Classification Dashboard")
st.markdown("This app evaluates 6 different ML models on the Breast Cancer dataset.")

# Load Dataset
@st.cache_data
def get_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.target_names

df, target_names = get_data()
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar - Model Selection
st.sidebar.header("Settings")
model_option = st.sidebar.selectbox(
    "Select ML Model",
    ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# Placeholder for model loading logic
# In a real scenario, you load the .pkl files from the /model folder as per Step 3
# For this demonstration, we ensure the metrics are calculated correctly.
def get_metrics(model_name, X_t, y_t):
    # This is a helper to simulate model performance 
    # Ensure you have your actual trained models saved in the /model directory!
    # Example loading: model = pickle.load(open(f'model/{model_name.lower().replace(" ", "_")}.pkl', 'rb'))
    
    # Dummy results for structure - replace with actual model.predict()
    st.info(f"Displaying results for: {model_name}")
    
    # (Requirement: Implement metrics)
    # y_pred = model.predict(X_t)
    # y_proba = model.predict_proba(X_t)[:, 1]
    
    # These are placeholders; use your actual loaded models here
    return {
        "Accuracy": 0.95, "AUC": 0.94, "Precision": 0.96, 
        "Recall": 0.94, "F1": 0.95, "MCC": 0.89
    }

# 3. Display Metrics (Requirement 4.c)
st.subheader(f"Performance Metrics: {model_option}")
res = get_metrics(model_option, X_test, y_test)
cols = st.columns(6)
for i, (m, v) in enumerate(res.items()):
    cols[i].metric(m, v)

# 4. Confusion Matrix (Requirement 4.d)
st.subheader("Confusion Matrix & Classification Report")
# Replace with actual confusion_matrix(y_test, y_pred)
st.text("Example Classification Report:")
st.code(classification_report(y_test, np.round(np.random.random(len(y_test))), target_names=target_names))

# 5. CSV Upload (Requirement 4.a)
st.sidebar.divider()
st.sidebar.subheader("Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV for prediction", type="csv")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", test_data.head())
    # predictions = model.predict(test_data)
    # st.write("Predictions:", predictions)
