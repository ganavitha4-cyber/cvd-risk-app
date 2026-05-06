import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Community Cardiovascular Risk Prediction Tool ❤️")

# -------- INPUTS --------
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

# -------- PREDICT --------
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
        'Covid_Severity': [
            1 if covid_severity == "Mild" else 
            2 if covid_severity == "Moderate" else 3
        ],
        'Vaccination_status': [1 if vaccine == "Yes" else 0]
    })

    # Prediction
    prediction = model.predict(input_data)[0]

    # Risk label
    if prediction < 10:
        label = "Low Risk"
    elif prediction < 20:
        label = "Medium Risk"
    else:
        label = "High Risk"

    st.success(f"Predicted Risk Score: {prediction}")
    st.success(f"Risk Category: {label}")