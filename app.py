import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

st.title("Community Cardiovascular Risk Prediction Tool ❤️")

# ---------------- FEATURE ALIGNMENT FUNCTION ----------------
def align_features(input_df, model):

    # Case 1: model has feature names (best case)
    if hasattr(model, "feature_names_in_"):
        model_features = model.feature_names_in_

        # Add missing columns
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder correctly
        input_df = input_df[model_features]

    else:
        # Case 2: fallback using feature count
        expected = model.n_features_in_

        while input_df.shape[1] < expected:
            input_df[f"extra_{input_df.shape[1]}"] = 0

        input_df = input_df.iloc[:, :expected]

    return input_df


# ---------------- INPUTS ----------------
age = st.number_input("Age", 18, 100)
gender = st.selectbox("Sex", ["Male", "Female"])

height = st.number_input("Height (cm)", 100, 250)
weight = st.number_input("Weight (kg)", 30, 200)
bmi = st.number_input("BMI", 10.0, 50.0)

systolic_bp = st.number_input("Systolic BP", 50, 300)
diastolic_bp = st.number_input("Diastolic BP", 40, 200)

hypertension = st.selectbox("Hypertension", ["Yes", "No"])
diabetes = st.selectbox("Diabetes", ["Yes", "No"])
ckd = st.selectbox("CKD", ["Yes", "No"])
smoking = st.selectbox("Smoking", ["Yes", "No"])

covid_hx = st.selectbox("COVID History", ["Yes", "No"])
covid_severity = st.selectbox("COVID Severity", ["Mild", "Moderate", "Severe"])
vaccine = st.selectbox("Vaccinated", ["Yes", "No"])


# ---------------- PREDICTION ----------------
if st.button("Predict Risk"):

    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [1 if gender == "Male" else 0],
        'Height_cm': [height],
        'Weight_kg': [weight],
        'BMI': [bmi],
        'Systolic_BP': [systolic_bp],
        'Diastolic_BP': [diastolic_bp],
        'Hypertension': [1 if hypertension == "Yes" else 0],
        'Diabetes': [1 if diabetes == "Yes" else 0],
        'CKD': [1 if ckd == "Yes" else 0],
        'Smoking_Status': [1 if smoking == "Yes" else 0],
        'Covid_Hx': [1 if covid_hx == "Yes" else 0],
        'Covid_Severity': 1 if covid_severity == "Mild"
            else 2 if covid_severity == "Moderate"
            else 3,
        'Vaccination_status': [1 if vaccine == "Yes" else 0]
    })

    try:
        # 🔥 AUTO FEATURE ALIGNMENT
        input_data = align_features(input_data, model)

        prediction = model.predict(input_data)[0]

        # ---------------- RISK LABEL ----------------
        if prediction < 10:
            label = "Low Risk"
        elif prediction < 20:
            label = "Medium Risk"
        else:
            label = "High Risk"

        st.success(f"Predicted Risk Score: {prediction}")
        st.success(f"Risk Category: {label}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")