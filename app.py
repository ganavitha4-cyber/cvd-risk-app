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

        # ---------------- SHAP EXPLAINABILITY ----------------
        st.write("## 🔍 AI Explanation (SHAP)")

        try:
            explainer = shap.Explainer(model)
        except:
            explainer = shap.TreeExplainer(model)

        shap_values = explainer(input_data)

        st.write("### Individual Prediction Breakdown")

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

        # Top features
        st.write("### 🔑 Most Influential Features")

        shap_df = pd.DataFrame({
            "Feature": input_data.columns,
            "Impact": shap_values.values[0]
        })

        shap_df["AbsImpact"] = np.abs(shap_df["Impact"])
        shap_df = shap_df.sort_values(by="AbsImpact", ascending=False)

        for i in range(min(5, len(shap_df))):
            row = shap_df.iloc[i]
            direction = "increases" if row["Impact"] > 0 else "decreases"
            st.write(f"• {row['Feature']} **{direction}** your risk")

    except Exception as e:
        st.error(f"Prediction failed: {e}")