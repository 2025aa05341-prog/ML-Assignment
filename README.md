# ML-Assignment

1. Problem Statement
This assignment focuses on building and evaluating multiple machine learning classification models, and deploying them using Streamlit.
You are required to:
	• Choose a public dataset
	• Train six classification models
	• Evaluate them using six metrics
	• Build an interactive Streamlit web application
	• Deploy it on Streamlit Community Cloud
	• Submit GitHub link + Live App link + BITS Lab Execution Screenshot
This project uses the Breast Cancer Wisconsin (Diagnostic) dataset for binary classification:
0 = Malignant, 1 = Benign

2. Dataset Description
Property	Details
Dataset	Breast Cancer Wisconsin (Diagnostic)
Source	sklearn.datasets.load_breast_cancer()
Instances	569 samples
Features	30 continuous features
Target Classes	0 (Malignant), 1 (Benign)
This dataset includes radius, perimeter, smoothness, texture, compactness, concavity, and other morphological features derived from cell nuclei images.

3. Models Used and Evaluation Metrics
The following six classification models were trained and evaluated:
	1. Logistic Regression
	2. Decision Tree Classifier
	3. k‑Nearest Neighbors (kNN)
	4. Naive Bayes (Gaussian NB)
	5. Random Forest (Ensemble)
	6. XGBoost (Ensemble)
Each model was evaluated using:
	• Accuracy
	• AUC
	• Precision
	• Recall
	• F1 Score
	• MCC (Matthews Correlation Coefficient)

4. Comparison Table of All Models
Metrics were computed using an 80‑20 train-test split with stratification.
Model	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.9825	0.9954	0.9861	0.9861	0.9861	0.9623
Decision Tree	0.9123	0.9157	0.9559	0.9028	0.9286	0.8174
kNN	0.9737	0.9884	0.9600	1.0000	0.9796	0.9442
Naive Bayes	0.9298	0.9868	0.9444	0.9444	0.9444	0.8492
Random Forest (Ensemble)	0.9474	0.9937	0.9583	0.9583	0.9583	0.8869
XGBoost (Ensemble)	0.9561	0.9950	0.9467	0.9861	0.9660	0.9058

5. Observations on Model Performance
Model	Observation
Logistic Regression	Shows best overall performance with 0.9825 accuracy and excellent AUC of 0.9954. Indicates dataset is nearly linearly separable.
Decision Tree	Lowest performer due to overfitting, shown by reduced accuracy (0.9123) and MCC (0.8174).
kNN	Extremely strong recall (1.0) with a high F1 score, meaning it catches all positive cases but has small false positives. Performs well after scaling.
Naive Bayes	Fast and simple but accuracy (0.9298) is lower due to its independence assumptions, although AUC is still high (0.9868).
Random Forest	Strong and stable performance across all metrics. Less overfitting than Decision Tree; good balance of bias and variance.
XGBoost	Among the top performers with AUC 0.9950, excellent recall & F1. Handles nonlinearities and feature interactions very well.

6. Project Structure
project-folder/
│-- app.py
│-- generate_models.py
│-- requirements.txt
│-- README.md
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
└── data/
    └── sample_test.csv

7. Streamlit App Features
✔ Upload test CSV
✔ Model selection dropdown
✔ Display evaluation metrics
✔ Confusion matrix
✔ Classification report
✔ Download predictions
✔ Optimized for speed & low memory usage
✔ Minimal dependency footprint for fast deployment

8. How to Run Locally
Install dependencies
pip install -r requirements.txt
Generate model PKL files
python generate_models.py
Run the Streamlit app
streamlit run app.py

9. Required Submission Links
Add these before uploading the final PDF:
Item	Link
GitHub Repository	paste your link here
Streamlit App (Live)	paste your link here
BITS Virtual Lab Screenshot	insert screenshot in PDF

10. Acknowledgements
	• Scikit‑learn
	• Streamlit
	• BITS WILP
	• Breast Cancer Wisconsin Dataset
