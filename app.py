# app.py (FAST, CLEAN, NO hard dependency on joblib)
# Streamlit app for ML Assignment‑2 — loads pickled models with a safe fallback.

import os # Added for local file pathing
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

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
# Page config
# ------------------------
st.set_page_config(page_title="ML Assignment 2 — Breast Cancer", layout="wide")
st.title("Machine Learning Assignment‑2 — Classification on Breast Cancer Dataset")
st.caption("Lightweight app optimized for Streamlit Community Cloud. No hard dependency on joblib.")

# ------------------------
# Constants
# ------------------------
MODEL_DIR = Path("model")
# Ensure model directory exists for local runs (though already handled in Colab setup)
if not MODEL_DIR.exists():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
    "Decision Tree": MODEL_DIR / "decision_tree.pkl",
    "kNN": MODEL_DIR / "knn.pkl",
    "Naive Bayes": MODEL_DIR / "naive_bayes.pkl",
    "Random Forest": MODEL_DIR / "random_forest.pkl",
    "XGBoost": MODEL_DIR / "xgboost.pkl",
}

# Optional import check for XGBoost (to avoid unpickle errors if lib missing)
try:
    import xgboost  # noqa: F401
    _XGB_AVAILABLE = True
except Exception:
    _XGB_AVAILABLE = False

# ------------------------
# Cached loaders
# ------------------------
@st.cache_resource(show_spinner=False)
def load_dataset():
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return X, y, list(ds.feature_names), list(ds.target_names)

# Robust artifact loader: try joblib → pickle
@st.cache_resource(show_spinner=False)
def load_artifact(path: Path):
    # Try joblib if available
    try:
        import joblib  # local import; if missing, fall through
        return joblib.load(path)
    except Exception:
        pass
    # Fallback to pickle
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)

# ------------------------
# Utilities
# ------------------------
def get_proba_or_score(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            return None
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except Exception:
            return None
    return None

# ------------------------
# Data preparation (single split)
# ------------------------
X, y, feature_names, target_names = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------
# Sidebar controls
# ------------------------
st.sidebar.header("Controls")

# Detect which model files exist
existing = {name: p for name, p in MODEL_FILES.items() if p.exists()}

# If xgboost lib is missing, drop XGBoost even if file exists (cannot unpickle)
if "XGBoost" in existing and not _XGB_AVAILABLE:
    del existing["XGBoost"]
    st.sidebar.warning("XGBoost library not available — skipping XGBoost model.")

if not existing:
    st.error(
        "No usable model artifacts found in ./model. Run `python generate_models.py` "
        "locally to produce the .pkl files, then redeploy."
    )
    # st.stop() # Commented out for notebook execution flow
    st.warning("Warning: Streamlit app will not function correctly without model files. Please ensure generate_models.py has been run.")
    # Exit gracefully in a notebook context if models are truly missing and st.stop() is not effective
    # You might want to uncomment st.stop() if running in a pure Streamlit environment
    # For Colab, we'll let it proceed but show the warning.

# Only proceed if existing models are found, otherwise model_name will be None
if existing:
    model_name = st.sidebar.selectbox("Select model", list(existing.keys()), index=0)

    # ------------------------
    # Load selected model (fast)
    # ------------------------
    model_path = existing[model_name]
    try:
        model = load_artifact(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {model_path}\nError: {e}")
        # st.stop() # Commented out for notebook execution flow
        st.warning(f"Warning: Model '{model_name}' could not be loaded. App functionality may be limited.")
        model = None # Set model to None if loading fails
else:
    model_name = None
    model = None

if model is not None: # Only proceed with evaluation if a model was successfully loaded
    # ------------------------
    # Evaluate on test split
    # ------------------------
    with st.spinner(f"Evaluating {model_name} ..."):
        y_pred = model.predict(X_test)
        y_score = get_proba_or_score(model, X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "AUC": (roc_auc_score(y_test, y_score) if y_score is not None else np.nan),
    }

    st.subheader("Evaluation Metrics (Test Split)")
    metrics_df = (
        pd.DataFrame([metrics]).T.reset_index().rename(columns={"index": "Metric", 0: "Value"})
    )
    metrics_df["Value"] = metrics_df["Value"].apply(
        lambda v: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
    )
    # Updated for deprecation warning
    st.dataframe(metrics_df, width='stretch')

    # Confusion matrix table (no heavy plotting)
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    # Updated for deprecation warning
    st.dataframe(cm_df, width='stretch')

    # Classification report
    st.subheader("Classification Report")
    report_txt = classification_report(y_test, y_pred, target_names=target_names)
    st.code(report_txt, language="text")

    # ------------------------
    # CSV upload for test data
    # ------------------------
    st.subheader("Upload a test CSV (optional)")
    st.caption(
        "Upload **only test data**. The CSV must have the same 30 feature columns "
        "as the sklearn dataset (names must match)."
    )

    col1, _ = st.columns(2)
    with col1:
        if st.button("Generate sample_test.csv (first 10 rows of X_test)"):
            sample = X_test.iloc[:10].copy()
            st.download_button(
                label="Download sample_test.csv",
                data=sample.to_csv(index=False).encode("utf-8"),
                file_name="sample_test.csv",
                mime="text/csv",
            )

    uploaded = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded is not None:
        try:
            test_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            # st.stop() # Commented out for notebook execution flow
            st.warning("Warning: Could not read uploaded CSV. Please check the file format.")

        # Validate columns
        missing = [c for c in feature_names if c not in test_df.columns]
        extra = [c for c in test_df.columns if c not in feature_names]

        if missing:
            st.error(f"Missing expected columns: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        else:
            if extra:
                st.warning(f"Extra columns will be ignored: {extra[:10]}{' ...' if len(extra) > 10 else ''}")
            test_df = test_df[feature_names]
            preds = model.predict(test_df)
            scores = get_proba_or_score(model, test_df)

            out = pd.DataFrame({"prediction": preds.astype(int)})
            if scores is not None:
                if hasattr(model, "predict_proba"):
                    out["prob_class_1"] = scores
                else:
                    out["score"] = scores

            # Updated for deprecation warning
            st.dataframe(out.head(20), width='stretch')
            st.download_button(
                "Download predictions",
                out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )