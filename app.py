import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="CVD Risk Prediction", layout="centered")

st.title("❤️ AI-Based Cardiovascular Risk Prediction")
st.write("Enter patient details based on clinical data:")

# ---------------- INPUTS ----------------
age = st.number_input("Age", 18, 100)

gender = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0

height = st.number_input("Height (cm)", 100, 220)
weight = st.number_input("Weight (kg)", 30, 200)

# BMI auto
bmi = weight / ((height / 100) ** 2)

sbp = st.number_input("Systolic BP", 80, 200)
dbp = st.number_input("Diastolic BP", 50, 150)

# Clinical Conditions
hypertension = st.selectbox("Hypertension", [0, 1])
diabetes = st.selectbox("Diabetes", [0, 1])
ckd = st.selectbox("Chronic Kidney Disease", [0, 1])
ra = st.selectbox("Rheumatoid Arthritis", [0, 1])
af = st.selectbox("Atrial Fibrillation", [0, 1])
migraine = st.selectbox("Migraine", [0, 1])
smi = st.selectbox("Severe Mental Illness", [0, 1])
sle = st.selectbox("SLE", [0, 1])
ed = st.selectbox("Erectile Dysfunction", [0, 1])
family_history = st.selectbox("Family History of CVD", [0, 1])

# Lifestyle
smoking = st.selectbox("Smoking Status", [0, 1])

# COVID Related
covid = st.selectbox("COVID History", [0, 1])

if covid == 1:
    covid_year = st.number_input("COVID Year", 2000, 2026)
    covid_severity = st.selectbox("COVID Severity", [1, 2, 3])
else:
    covid_year = 0
    covid_severity = 0

vaccination = st.selectbox("Vaccination Status", [0, 1])

# ---------------- DATAFRAME ----------------
input_data = pd.DataFrame([[
    age, gender, height, weight, bmi,
    sbp, dbp,
    hypertension, diabetes, ckd,
    ra, af, migraine, smi, sle,
    ed, family_history, smoking,
    covid, covid_year, covid_severity, vaccination,
0
]], columns=[
    'Age', 'Gender', 'Height_cm', 'Weight_kg', 'BMI',
    'Systolic_BP', 'Diastolic_BP',
    'Hypertension', 'Diabetes', 'CKD',
    'Rheumatoid_Arthritis', 'Atrial_Fibrillation',
    'Migraine', 'Severe_Mental_Illness', 'SLE',
    'Erectile_Dysfunction', 'Family_Hx_CVD',
    'Smoking_Status', 'COVID_Hx', 'COVID_Year',
    'COVID_Severity', 'Vaccination_Status',
    'Extra1'
])

# ---------------- PREDICT ----------------
if st.button("Predict Risk"):
    try:
        pred = model.predict(input_data)[0]

        try:
            prob = model.predict_proba(input_data)[0][1] * 100
        except:
            prob = float(pred)

        # CATEGORY
        if prob < 10:
            category = "🟢 Low Risk"
        elif prob < 20:
            category = "🟡 Moderate Risk"
        else:
            category = "🔴 High Risk"

        st.subheader(f"Risk Score: {prob:.2f}%")
        st.subheader(f"Category: {category}")

        # ---------------- CONTRIBUTING FACTORS ----------------
        st.write("### 📊 Key Contributing Factors")

        factors = []

        if age > 55:
            factors.append("Older age increases CVD risk")

        if bmi > 30:
            factors.append("Obesity (High BMI)")

        if sbp > 140 or dbp > 90:
            factors.append("High Blood Pressure")

        if hypertension == 1:
            factors.append("Existing Hypertension")

        if diabetes == 1:
            factors.append("Diabetes")

        if ckd == 1:
            factors.append("Chronic Kidney Disease")

        if af == 1:
            factors.append("Atrial Fibrillation")

        if smoking == 1:
            factors.append("Smoking habit")

        if family_history == 1:
            factors.append("Family History of CVD")

        if covid_severity >= 2:
            factors.append("Severe COVID history")

        if len(factors) == 0:
            st.success("No major contributing risk factors")
        else:
            for f in factors:
                st.warning(f)

        # ---------------- RANGES ----------------
        st.write("### 📈 Clinical Risk Ranges")

        st.info("""
        Age:
        - Low: < 40
        - Moderate: 40–55
        - High: > 55

        BMI:
        - Normal: 18.5–24.9
        - Overweight: 25–29.9
        - Obese: ≥30

        Blood Pressure:
        - Normal: <120/80
        - Elevated: 120–139 / 80–89
        - High: ≥140/90
        """)

               # ---------------- SHAP EXPLANATION ----------------
st.write("## 🔍 AI Explanation (SHAP)")

try:
    import shap
    import matplotlib.pyplot as plt

    # ✅ Ensure input is DataFrame with correct column names
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data, columns=feature_names)

    # ---------------- CREATE EXPLAINER ----------------
    try:
        explainer = shap.Explainer(model, input_data)
        shap_values = explainer(input_data)

        values = shap_values.values[0]
        base_values = shap_values.base_values[0]

    except Exception:
        # Fallback for tree models (RF, XGBoost, etc.)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_data)

        if isinstance(shap_vals, list):  # classification
            values = shap_vals[1][0]
            base_values = explainer.expected_value[1]
        else:  # regression
            values = shap_vals[0]
            base_values = explainer.expected_value

    # ---------------- CREATE EXPLANATION OBJECT ----------------
    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=input_data.iloc[0],
        feature_names=input_data.columns.tolist()
    )

    # ---------------- WATERFALL PLOT ----------------
    st.write("### 📊 Feature Contribution")

    fig1, ax1 = plt.subplots()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig1)
    plt.close(fig1)

    # ---------------- BAR PLOT ----------------
    st.write("### 🔝 Feature Importance")

    fig2, ax2 = plt.subplots()
    shap.plots.bar(explanation, show=False)
    st.pyplot(fig2)
    plt.close(fig2)

except Exception as e:
    st.warning(f"⚠️ SHAP explanation not available: {e}")

    # ✅ Safe fallback
    st.write("### 📌 Input Summary")
    for col, val in zip(input_data.columns, input_data.iloc[0]):
        st.write(f"**{col}**: {val}")