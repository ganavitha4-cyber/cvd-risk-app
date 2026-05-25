import streamlit as st
import pandas as pd
import pickle

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="CVD Risk Predictor", layout="centered")

st.title("❤️ AI-Based Cardiovascular Risk Prediction")
st.write("Enter patient details below:")

# ---------------- INPUTS ----------------
age = st.number_input("Age", 18, 100, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", 10.0, 50.0)
systolic_bp = st.number_input("Systolic Blood Pressure", 80, 200)

diabetes = st.selectbox("Diabetes", ["No", "Yes"])
smoking = st.selectbox("Smoking", ["No", "Yes"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])

# ---------------- PREDICTION ----------------
if st.button("Predict Risk"):

    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [1 if gender == "Male" else 0],
            'BMI': [bmi],
            'Systolic_BP': [systolic_bp],
            'Diabetes': [1 if diabetes == "Yes" else 0],
            'Smoking': [1 if smoking == "Yes" else 0],
            'Hypertension': [1 if hypertension == "Yes" else 0]
        })

        # ---------------- HANDLE FEATURE MISMATCH ----------------
        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in input_data.columns:
                    input_data[col] = 0

            input_data = input_data[model.feature_names_in_]

        # ---------------- PREDICT ----------------
        if hasattr(model, "predict_proba"):
            prediction = model.predict_proba(input_data)[0][1] * 100
        else:
            prediction = model.predict(input_data)[0]

        # ---------------- CATEGORY ----------------
        if prediction < 10:
            category = "🟢 Low Risk"
        elif prediction < 20:
            category = "🟡 Moderate Risk"
        else:
            category = "🔴 High Risk"

        st.success(f"Predicted Risk Score: {prediction:.2f}%")
        st.success(f"Risk Category: {category}")

        # ---------------- FACTORS ----------------
        st.subheader("🔍 Key Contributing Factors")

        factors = []

        if age > 50:
            factors.append("Age > 50 → higher cardiovascular strain")

        if bmi > 25:
            factors.append("BMI > 25 → overweight/obesity risk")

        if systolic_bp > 140:
            factors.append("Systolic BP > 140 → hypertension risk")

        if diabetes == "Yes":
            factors.append("Diabetes → damages blood vessels")

        if smoking == "Yes":
            factors.append("Smoking → reduces oxygen & damages arteries")

        if hypertension == "Yes":
            factors.append("Existing hypertension → increases heart load")

        if factors:
            for f in factors:
                st.write("•", f)
        else:
            st.write("No major contributing factors detected")

        # ---------------- WHY ----------------
        st.subheader("🧠 Why this prediction?")
        st.info("""
This prediction is generated using an AI model trained on cardiovascular risk factors.

Risk increases when:
- Age increases
- BMI is high
- Blood pressure is elevated
- Lifestyle risks (smoking, diabetes) are present

The model combines all these factors to estimate overall cardiovascular risk.
        """)

        # ---------------- RISK RANGES ----------------
        st.subheader("📊 Risk Classification")

        st.markdown("""
- 🟢 **Low Risk:** 0 – 10%  
- 🟡 **Moderate Risk:** 10 – 20%  
- 🔴 **High Risk:** > 20%  
        """)

    except Exception as e:
        st.error("❌ Prediction failed")
        st.write(e)