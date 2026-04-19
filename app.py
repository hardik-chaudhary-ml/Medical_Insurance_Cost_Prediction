import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model/insurance_model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Insurance Predictor", page_icon="🔥", layout="centered")

st.markdown("""
<style>
body {
    background: linear-gradient(to right, #667eea, #764ba2);
}
.main {
    background-color: rgba(255,255,255,0.95);
    padding: 25px;
    border-radius: 15px;
}
h1 {
    text-align: center;
    color: #333;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🔥 Insurance Cost Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict your medical insurance cost using Machine Learning</p>", unsafe_allow_html=True)

st.subheader("📋 Enter Details")

# Layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 25)
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)

with col2:
    children = st.slider("Children", 0, 5, 0)

# Inputs
sex = st.radio("Sex", ["Male", "Female"])
sex_encoded = 1 if sex == "Male" else 0

smoker = st.radio("Smoker", ["Yes", "No"])
smoker_encoded = 1 if smoker == "Yes" else 0

region = st.selectbox("Region", ["southeast", "southwest", "northwest", "northeast"])
region_encoded = {
    "southeast": 0,
    "southwest": 1,
    "northwest": 2,
    "northeast": 3
}[region]

if st.button("🚀 Predict Insurance Cost"):

    input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Insurance Cost: ${round(prediction, 2)}")

    
    st.subheader("📈 Cost vs Age Trend")

    ages = np.arange(18, 65)
    costs = []

    for a in ages:
        temp_input = np.array([[a, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
        cost = model.predict(temp_input)[0]
        costs.append(cost)

    fig, ax = plt.subplots()
    ax.plot(ages, costs, marker='o')
    ax.set_xlabel("Age")
    ax.set_ylabel("Predicted Cost")
    ax.set_title("Insurance Cost vs Age")

    st.pyplot(fig)

    st.subheader("📊 Insights")

    if smoker_encoded == 1:
        st.warning("⚠️ Smoking significantly increases insurance cost!")

    if bmi > 30:
        st.warning("⚠️ High BMI may increase insurance cost!")

    if age > 50:
        st.info("ℹ️ Higher age may lead to higher premiums.")

    if children > 2:
        st.info("ℹ️ More dependents can increase cost slightly.")