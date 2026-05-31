# Bank-Term-Deposit-Prediction
Bank Term Deposit Subscription Prediction using Machine Learning and Deep Learning

## Project Overview

This project predicts whether a bank customer will subscribe to a term deposit based on demographic, financial, and campaign-related information.

The objective is to help banks identify potential customers and improve marketing campaign effectiveness.

---

## Dataset

Dataset: Bank Marketing Dataset

Features include:

- Age
- Job
- Marital Status
- Education
- Balance
- Housing Loan
- Personal Loan
- Contact Type
- Campaign Information
- Previous Campaign Outcome

Target Variable:

- y = Yes (Subscribed)
- y = No (Not Subscribed)

---

## Project Workflow

### 1. Data Preprocessing

- Missing value handling
- Ordinal Encoding
- One-Hot Encoding
- Feature Scaling

### 2. Feature Engineering

Created interaction features:

- age_balance
- duration_balance
- age_education
- duration_education

### 3. Models Evaluated

- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
- SVM
- Naive Bayes
- LightGBM
- CatBoost
- Deep Learning (ANN)
- Stacking Classifier

### 4. Model Evaluation

Metrics Used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### 5. Explainability

- Feature Importance
- SHAP Analysis

### 6. Cross Validation

- Stratified K-Fold Cross Validation

---

## Final Model

### LightGBM

Performance:

- Accuracy: 90.90%
- ROC-AUC: 0.9346

Cross Validation:

- Mean ROC-AUC: 0.9360
- Standard Deviation: 0.0032

LightGBM was selected as the final model due to its strong predictive performance, stability, and deployment suitability.

---

## Deployment

A Streamlit application was developed to allow users to enter customer information and receive subscription predictions in real time.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- LightGBM
- XGBoost
- CatBoost
- TensorFlow / Keras
- SHAP
- Streamlit

---

## Repository Structure

```

Bank-Term-Deposit-Prediction/
│
├── bank_term.ipynb
├── Bank_Marketing_Deep_Learning.ipynb
├── app.py
├── lightgbm_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
└── README.md

```

---

## Author

Shamela K

