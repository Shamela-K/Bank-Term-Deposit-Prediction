import streamlit as st
import pandas as pd
import joblib

# ==========================================

# PAGE CONFIG

# ==========================================

st.set_page_config(
page_title="Bank Term Deposit Prediction",
page_icon="🏦",
layout="wide"
)

# ==========================================

# LOAD MODEL

# ==========================================

model = joblib.load("lightgbm_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ==========================================

# HEADER

# ==========================================

st.title("🏦 Bank Term Deposit Prediction Dashboard")

st.markdown(
"""
Predict whether a customer will subscribe to a bank term deposit using a
Machine Learning model trained on the UCI Bank Marketing Dataset.
"""
)

# ==========================================

# SIDEBAR

# ==========================================

st.sidebar.title("📌 Project Information")

st.sidebar.success("Final Model: LightGBM")

st.sidebar.markdown("""

### Techniques Used

✅ Feature Engineering

✅ Class Imbalance Analysis

✅ SMOTE

✅ LightGBM

✅ SHAP Explainability

✅ Stratified K-Fold Validation

✅ Streamlit Deployment
""")

# ==========================================

# MODEL METRICS

# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
  st.metric("Accuracy", "90.90%")

with col2:
  st.metric("ROC-AUC", "93.46%")

with col3:
  st.metric("Features Used", "44")

# ==========================================

# MODEL COMPARISON

# ==========================================

st.subheader("📊 Model Comparison Leaderboard")

comparison = pd.DataFrame({
"Rank":[1,2,3,4,5,6],
"Model":[
"Blending",
"Stacking",
"LightGBM",
"CatBoost",
"XGBoost",
"Random Forest"
],
"ROC-AUC":[
0.9353,
0.9347,
0.9346,
0.9325,
0.9277,
0.9232
]
})

st.dataframe(comparison, width="stretch")

st.markdown("---")

# ==========================================

# CUSTOMER INPUT

# ==========================================

st.subheader("🧾 Customer Information")

col1, col2 = st.columns(2)

with col1:


 age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

job = st.selectbox(
    "Job",
    [
        "admin.","blue-collar","entrepreneur",
        "housemaid","management","retired",
        "self-employed","services","student",
        "technician","unemployed","unknown"
    ]
)

marital = st.selectbox(
    "Marital Status",
    ["divorced","married","single"]
)

education = st.selectbox(
    "Education",
    ["unknown","primary","secondary","tertiary"]
)

default = st.selectbox(
    "Default Credit",
    ["no","yes"]
)

balance = st.number_input(
    "Balance",
    value=1000
)

housing = st.selectbox(
    "Housing Loan",
    ["no","yes"]
)


with col2:


 loan = st.selectbox(
    "Personal Loan",
    ["no","yes"]
)

contact = st.selectbox(
    "Contact Type",
    ["cellular","telephone","unknown"]
)

month = st.selectbox(
    "Month",
    [
        "jan","feb","mar","apr","may","jun",
        "jul","aug","sep","oct","nov","dec"
    ]
)

day = st.number_input(
    "Last Contact Day",
    min_value=1,
    max_value=31,
    value=15
)

duration = st.number_input(
    "Call Duration (seconds)",
    min_value=0,
    value=300
)

campaign = st.number_input(
    "Campaign Contacts",
    min_value=1,
    value=1
)

pdays = st.number_input(
    "Days Since Previous Contact",
    value=-1
)

previous = st.number_input(
    "Previous Contacts",
    min_value=0,
    value=0
)

poutcome = st.selectbox(
    "Previous Campaign Outcome",
    ["failure","other","success","unknown"]
)


# ==========================================

# PREDICTION

# ==========================================

if st.button("🔮 Predict Subscription"):
    education_map = {
        "unknown":0,
        "primary":1,
        "secondary":2,
        "tertiary":3
    }
    edu_num = education_map[education]

    # Feature Engineering

    age_balance = age * balance
    duration_balance = duration * balance
    age_education = age * edu_num
    duration_education = duration * edu_num

    input_df = pd.DataFrame({
        "age":[age],
        "education":[edu_num],
        "balance":[balance],
        "day":[day],
        "duration":[duration],
        "campaign":[campaign],
        "pdays":[pdays],
        "previous":[previous],
        "age_balance":[age_balance],
        "duration_balance":[duration_balance],
        "age_education":[age_education],
        "duration_education":[duration_education]
    })

    dummy_cols = {

        f"job_{job}":1,
        f"marital_{marital}":1,

        "default_yes":1 if default=="yes" else 0,
        "housing_yes":1 if housing=="yes" else 0,
        "loan_yes":1 if loan=="yes" else 0,

        f"contact_{contact}":1,
        f"month_{month}":1,
        f"poutcome_{poutcome}":1
    }

    for col,val in dummy_cols.items():
        input_df[col] = val

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.success("✅ Customer is likely to subscribe")
        st.balloons()
    else:
        st.error("❌ Customer is unlikely to subscribe")

    st.subheader("📈 Subscription Probability")

    st.progress(float(probability))

    st.metric(
        "Probability",
        f"{probability*100:.2f}%"
    )

    if probability >= 0.80:
        st.success("🔥 Very High Chance")

    elif probability >= 0.60:
        st.info("👍 Good Chance")

    elif probability >= 0.40:
        st.warning("⚠ Moderate Chance")

    else:
        st.error("❌ Low Chance")

    st.subheader("💡 Business Insight")

    if duration > 200:
        st.info(
            "Long call duration generally increases the likelihood of subscription."
        )

    if poutcome == "success":
        st.success(
            "Previous successful campaign outcome strongly improves conversion probability."
        )

# ==========================================

# FEATURE IMPORTANCE

# ==========================================

st.markdown("---")

st.subheader("🔍 Top Feature Importance")

st.image(
"feature_importance.png",
caption="Top 10 Important Features from LightGBM"
)

# ==========================================

# PROJECT SUMMARY

# ==========================================

st.markdown("---")

st.subheader("📋 Project Summary")

st.write("""
This project predicts whether a customer will subscribe to a bank term deposit.

### Techniques Used

* Feature Engineering
* Class Imbalance Analysis
* SMOTE
* LightGBM
* SHAP Explainability
* Stratified K-Fold Cross Validation
* Streamlit Deployment

### Final Selected Model

LightGBM

### Performance

* Accuracy: 90.90%
* ROC-AUC: 93.46%

The deployed model helps marketing teams identify customers with a high probability of subscribing to term deposits, enabling more efficient campaign targeting.
""")
