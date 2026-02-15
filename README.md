# Machine Learning Assignment-2

Name: Yella Sharath
Student ID: 2025AA05341
Email: 2025aa05341@wilp.bits-pilani.ac.in

# 1. Problem Statement
This assignment implements a complete end-to-end machine learning classification workflow for predicting whether a breast mass is Malignant or Benign using diagnostic measurements from digitized FNA tissue samples.
Key Workflow Components:
	• Training six machine learning models
	• Comprehensive model evaluation using standard metrics
	• Deployment of interactive Streamlit application
	• Clear systematic presentation of results
This is a high-impact medical classification task where model reliability, recall, and interpretability are critical.

# 2. Dataset Description
Breast Cancer Wisconsin (Diagnostic) Dataset
Property	Details
Total Instances	569
Total Features	30 numerical
Target Classes	Malignant (0), Benign (1)
Domain	Medical Diagnostics
Source	scikit-learn built-in dataset
The 30 features represent computed measurements of cell nuclei from breast mass images obtained via FNA biopsy.

# 3. Model Performance Comparison

| Model                 | Accuracy |   AUC   | Precision | Recall |  F1 Score |   MCC   |
|-----------------------|----------|---------|-----------|--------|-----------|---------|
| Logistic Regression   | 0.9825   | 0.9954  | 0.9861    | 0.9861 | 0.9861    | 0.9623  |
| Decision Tree         | 0.9123   | 0.9157  | 0.9559    | 0.9028 | 0.9286    | 0.8174  |
| k-Nearest Neighbors   | 0.9737   | 0.9884  | 0.9600    | 1.0000 | 0.9796    | 0.9442  |
| Naive Bayes           | 0.9298   | 0.9868  | 0.9444    | 0.9444 | 0.9444    | 0.8492  |
| Random Forest         | 0.9474   | 0.9937  | 0.9583    | 0.9583 | 0.9583    | 0.8869  |
| XGBoost               | 0.9561   | 0.9950  | 0.9467    | 0.9861 | 0.9660    | 0.9058  |

# 4. Key Observations & Model Analysis

| Model                 | Key Observation & Analysis |
|-----------------------|----------------------------|
| Logistic Regression   | **Top Performer:** Achieved the highest overall accuracy and MCC. Indicates that the dataset's features are largely linearly separable after standard scaling. |
| Decision Tree         | **Lowest Consistency:** Interpretable but lowest performer. Lacked pruning → mild overfitting → reduced robustness compared to ensemble models. |
| k-Nearest Neighbors   | **Highest Clinical Safety:** Delivered **perfect Recall (1.0000)**, ensuring zero malignant cases missed—critical in medical diagnostics. |
| Naive Bayes           | **Efficiency over Precision:** Extremely fast training/inference. Lower accuracy due to feature‑independence assumption, but maintained a strong AUC score. |
| Random Forest         | **Balanced Ensemble:** High stability. Averaging multiple trees greatly reduces variance and improves over the baseline Decision Tree. |
| XGBoost               | **High Reliability:** Gradient boosting achieved excellent balance of precision and recall, yielding one of the highest AUC scores (0.9950). |

	
# 5. Streamlit App Features
	•  Model Selection Dropdown
	•  Evaluation Metrics Table
	•  Color Confusion Matrix
	•  Classification Report
	•  ROC & PR Curves
	•  Model Leaderboard
	•  Adjustable Threshold
	•  CSV Batch Upload
	•  Download Predictions
