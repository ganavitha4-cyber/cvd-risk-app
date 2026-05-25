import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("❤️ AI-Based Cardiovascular Risk Prediction")
st.write("Enter patient details below:")

# -------- INPUTS --------
age = st.number_input("Age", 18, 100)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", 10.0, 50.0)
systolic_bp = st.number_input("Systolic BP", 50, 300)

diabetes = st.selectbox("Diabetes", ["Yes", "No"])
smoking = st.selectbox("Smoking", ["Yes", "No"])
hypertension = st.selectbox("Hypertension", ["Yes", "No"])

# -------- PREDICTION --------
if st.button("Predict Risk"):

    try:
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [1 if gender == "Male" else 0],
            'BMI': [bmi],
            'Systolic_BP': [systolic_bp],
            'Diabetes': [1 if diabetes == "Yes" else 0],
            'Smoking': [1 if smoking == "Yes" else 0],
            'Hypertension': [1 if hypertension == "Yes" else 0]
        })

        # Fix feature mismatch
        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in input_data.columns:
                    input_data[col] = 0

            input_data = input_data[model.feature_names_in_]

        prediction = model.predict(input_data)[0]

        # -------- CATEGORY --------
        if prediction < 10:
            category = "🟢 Low Risk"
        elif prediction < 20:
            category = "🟡 Moderate Risk"
        else:
            category = "🔴 High Risk"

        st.success(f"Predicted Risk Score: {prediction:.2f}%")
        st.success(f"Risk Category: {category}")

        # -------- FACTOR CONTRIBUTION --------
        st.subheader("🔍 Key Contributing Factors")

        factors = []

        if age > 50:
            factors.append("Higher age increases cardiovascular risk")

        if bmi > 25:
            factors.append("High BMI (overweight/obesity) increases risk")

        if systolic_bp > 140:
            factors.append("High blood pressure significantly increases risk")

        if diabetes == "Yes":
            factors.append("Diabetes is a major risk factor for CVD")

        if smoking == "Yes":
            factors.append("Smoking damages blood vessels and increases risk")

        if hypertension == "Yes":
            factors.append("Hypertension contributes to heart disease")

        if len(factors) == 0:
            st.write("No major risk factors detected")

        else:
            for f in factors:
                st.write("•", f)

        # -------- WHY (INTERPRETATION) --------
        st.subheader("🧠 Why this prediction?")

        st.write("""
        The model analyzes multiple clinical parameters such as age, BMI,
        blood pressure, and comorbidities. Higher values in these factors
        increase the probability of cardiovascular disease.
        """)

        # -------- RANGES --------
        st.subheader("📊 Risk Classification Range")

        st.write("""
        - 🟢 Low Risk: 0% – 10%
        - 🟡 Moderate Risk: 10% – 20%
        - 🔴 High Risk: > 20%
        """)

    except Exception as e:
        st.error("Prediction failed")
        st.write(e)