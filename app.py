import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("lightgbm_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Bank Term Deposit Prediction")

st.title("🏦 Bank Term Deposit Prediction")

st.write("Predict whether a customer will subscribe to a term deposit.")
st.subheader("Customer Summary")
# =========================
# User Inputs
# =========================

age = st.number_input("Age", min_value=18, max_value=100, value=35)

job = st.selectbox(
    "Job",
    [
        "admin.","blue-collar","entrepreneur","housemaid",
        "management","retired","self-employed",
        "services","student","technician",
        "unemployed","unknown"
    ]
)

marital = st.selectbox(
    "Marital Status",
    ["divorced", "married", "single"]
)

education = st.selectbox(
    "Education",
    ["unknown", "primary", "secondary", "tertiary"]
)

default = st.selectbox(
    "Default Credit",
    ["no", "yes"]
)

balance = st.number_input(
    "Balance",
    value=1000
)

housing = st.selectbox(
    "Housing Loan",
    ["no", "yes"]
)

loan = st.selectbox(
    "Personal Loan",
    ["no", "yes"]
)

contact = st.selectbox(
    "Contact Type",
    ["cellular", "telephone", "unknown"]
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
    ["failure", "other", "success", "unknown"]
)

# =========================
# Prediction Button
# =========================

if st.button("Predict"):

    # Education ordinal encoding
    education_map = {
        "unknown": 0,
        "primary": 1,
        "secondary": 2,
        "tertiary": 3
    }

    edu_num = education_map[education]

    # Feature engineering
    age_balance = age * balance
    duration_balance = duration * balance
    age_education = age * edu_num
    duration_education = duration * edu_num

    # Base dataframe
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

    # One-hot columns used during training

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

    # Add missing columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Correct order
    input_df = input_df[feature_columns]

    # Prediction
    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Customer is likely to subscribe")
    else:
        st.error("❌ Customer is unlikely to subscribe")

    st.metric(
        "Subscription Probability",
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