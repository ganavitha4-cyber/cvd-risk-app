import streamlit as st
import pandas as pd
import pickle

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

st.title("❤️ AI-Based Cardiovascular Risk Prediction")

st.write("Enter patient details below:")

# ---------------- INPUTS ----------------
age = st.number_input("Age", 18, 100)
gender = st.selectbox("Gender", ["Male", "Female"])

height = st.number_input("Height (cm)", 100, 220)
weight = st.number_input("Weight (kg)", 30, 150)
bmi = st.number_input("BMI", 10.0, 50.0)

systolic_bp = st.number_input("Systolic BP", 80, 200)
diastolic_bp = st.number_input("Diastolic BP", 50, 130)

htn = st.selectbox("Hypertension (HTN)", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
ckd = st.selectbox("Chronic Kidney Disease", ["No", "Yes"])

ra = st.selectbox("Rheumatoid Arthritis", ["No", "Yes"])
af = st.selectbox("Atrial Fibrillation", ["No", "Yes"])
migraine = st.selectbox("Migraine", ["No", "Yes"])
mental = st.selectbox("Severe Mental Illness", ["No", "Yes"])
sle = st.selectbox("SLE", ["No", "Yes"])
ed = st.selectbox("Erectile Dysfunction", ["No", "Yes"])

family = st.selectbox("Family History of CVD", ["No", "Yes"])
smoking = st.selectbox("Smoking Status", ["No", "Yes"])

covid = st.selectbox("COVID History", ["No", "Yes"])
covid_year = st.number_input("COVID Year", 2019, 2025)
covid_severity = st.selectbox("COVID Severity", ["Mild", "Moderate", "Severe"])

vaccine = st.selectbox("Vaccination Status", ["No", "Yes"])

# ---------------- PREDICTION ----------------
if st.button("Predict Risk"):

    try:
        input_data = pd.DataFrame({

            'age': [age],
            'gender': [1 if gender == "Male" else 0],
            'height_cm': [height],
            'weight_kg': [weight],
            'BMI': [bmi],

            'systolic_bp': [systolic_bp],
            'diastolic_bp': [diastolic_bp],

            'HTN': [1 if htn == "Yes" else 0],
            'Diabetes': [1 if diabetes == "Yes" else 0],
            'CKD': [1 if ckd == "Yes" else 0],

            'Rheumatoid_Arthritis': [1 if ra == "Yes" else 0],
            'Atrial_fibrillation': [1 if af == "Yes" else 0],
            'Migraine': [1 if migraine == "Yes" else 0],
            'Severe_mental_illness': [1 if mental == "Yes" else 0],
            'SLE': [1 if sle == "Yes" else 0],
            'Erectile_dysfunction': [1 if ed == "Yes" else 0],

            'Family_Hx_CVD': [1 if family == "Yes" else 0],
            'Smoking_Status': [1 if smoking == "Yes" else 0],

            'COVID_Hx': [1 if covid == "Yes" else 0],
            'COVID_Year': [covid_year],

            'Vaccination_Status': [1 if vaccine == "Yes" else 0]
        })

        # Encode COVID Severity
        input_data['COVID_Severity_Moderate'] = 1 if covid_severity == "Moderate" else 0
        input_data['COVID_Severity_Severe'] = 1 if covid_severity == "Severe" else 0

        # ---------------- FIX FEATURE COUNT ----------------
        while input_data.shape[1] < 23:
            input_data[f'extra_{input_data.shape[1]}'] = 0

        input_data = input_data.values

        # ---------------- PREDICT ----------------
        prediction = model.predict(input_data)[0]

        # ---------------- CATEGORY ----------------
        if prediction < 10:
            category = "🟢 Low Risk"
        elif prediction < 20:
            category = "🟡 Moderate Risk"
        else:
            category = "🔴 High Risk"

        st.success(f"Predicted Risk Score: {prediction:.2f}")
        st.success(f"Risk Category: {category}")

# ---------------- RISK RANGE ----------------
st.subheader("📊 Risk Classification")
st.write("""
🟢 Low Risk: 0 – 10%  
🟡 Moderate Risk: 10 – 20%  
🔴 High Risk: > 20%  
""")

# ---------------- FACTORS ----------------
st.subheader("🔍 Key Contributing Factors")

factors = []

if age > 50:
    factors.append("Age > 50 → arteries stiffen, increasing heart workload")

if bmi > 25:
    factors.append("BMI > 25 → obesity increases cardiovascular strain")

if systolic_bp > 140:
    factors.append("High systolic BP → damages blood vessels")

if diastolic_bp > 90:
    factors.append("High diastolic BP → increases heart pressure")

if htn == "Yes":
    factors.append("Hypertension → major risk for heart disease")

if diabetes == "Yes":
    factors.append("Diabetes → causes vascular damage")

if ckd == "Yes":
    factors.append("CKD → affects blood pressure & heart function")

if smoking == "Yes":
    factors.append("Smoking → reduces oxygen & damages arteries")

if family == "Yes":
    factors.append("Family history → genetic predisposition")

if covid == "Yes":
    factors.append("COVID history → possible cardiovascular complications")

if covid_severity == "Severe":
    factors.append("Severe COVID → increases long-term heart risk")

if vaccine == "No":
    factors.append("No vaccination → higher complication probability")

# Show factors
if factors:
    for f in factors:
        st.write("•", f)
else:
    st.write("No major contributing factors detected")

# ---------------- WHY ----------------
st.subheader("🧠 Why this prediction?")
st.info("""
This prediction is generated using an AI model trained on multiple clinical factors.

Risk increases when:
- Age is higher
- Blood pressure is elevated
- BMI indicates overweight/obesity
- Chronic diseases are present
- Lifestyle risks like smoking exist

The model combines all these inputs to estimate overall cardiovascular risk.
""")