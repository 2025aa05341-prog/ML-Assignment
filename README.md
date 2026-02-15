Machine Learning Assignment-2
Student Details
Name: Yella Sharath
Student ID: 2025AA05341
Email: 2025aa05341@wilp.bits-pilani.ac.in
Date: 15 February 2026
1. Problem Statement
This assignment implements a complete end-to-end machine learning classification workflow for predicting whether a breast mass is Malignant or Benign using diagnostic measurements from digitized FNA tissue samples.
Key Workflow Components:
	• Training six machine learning models
	• Comprehensive model evaluation using standard metrics
	• Deployment of interactive Streamlit application
	• Clear systematic presentation of results
This is a high-impact medical classification task where model reliability, recall, and interpretability are critical.
2. Dataset Description
Breast Cancer Wisconsin (Diagnostic) Dataset
Property	Details
Total Instances	569
Total Features	30 numerical
Target Classes	Malignant (0), Benign (1)
Domain	Medical Diagnostics
Source	scikit-learn built-in dataset
The 30 features represent computed measurements of cell nuclei from breast mass images obtained via FNA biopsy.
3. Submission Links
Requirement	Link
GitHub Repository	https://github.com/2025aa05341-prog/ML-Assignment
Live Streamlit App	https://ml-assignment-39ztzdnzedok8mxukjf9cs.streamlit.app/
4. Lab Execution Screenshots
	Place your BITS Virtual Lab screenshots in /screenshots/ folder
5. Model Performance Comparison
Model	Accuracy	AUC	Precision	Recall	F1-Score	MCC
Logistic Regression	0.9825	0.9954	0.9861	0.9861	0.9861	0.9623
Decision Tree	0.9123	0.9157	0.9559	0.9028	0.9286	0.8174
k-Nearest Neighbors	0.9737	0.9884	0.9600	1.0000	0.9796	0.9442
Naive Bayes	0.9298	0.9868	0.9444	0.9444	0.9444	0.8492
Random Forest	0.9474	0.9937	0.9583	0.9583	0.9583	0.8869
XGBoost	0.9561	0.9950	0.9467	0.9861	0.9660	0.9058
						
6. Key Observations & Model Analysis
Model	Key Observations
Logistic Regression	Top Performer: Highest accuracy & MCC. Linearly separable after scaling.
Decision Tree	Lowest consistency. Prone to overfitting without pruning.
k-NN	Perfect Recall (1.0): Zero malignant cases missed - clinically critical.
Naive Bayes	Fast training, strong AUC despite independence assumptions.
Random Forest	Balanced ensemble outperforms standalone Decision Tree.
XGBoost	Excellent AUC (0.9950), captures non-linear interactions.
	
7. Streamlit App Features
	•  Model Selection Dropdown
	•  Evaluation Metrics Table
	•  Color Confusion Matrix
	•  Classification Report
	•  ROC & PR Curves
	•  Model Leaderboard
	•  Adjustable Threshold
	•  CSV Batch Upload
	•  Download Predictions
