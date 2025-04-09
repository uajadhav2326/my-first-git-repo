import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("Lung Cancer Risk Prediction App")

st.write("Enter the details below to predict lung cancer risk.")

# Define input fields (update based on your dataset's features)
age = st.slider("Age", 18, 100, 30)
smoking = st.selectbox("Do you smoke?", ("Yes", "No"))
anxiety = st.selectbox("Do you have anxiety?", ("Yes", "No"))
chronic_disease = st.selectbox("Do you have any chronic disease?", ("Yes", "No"))
fatigue = st.selectbox("Do you experience fatigue?", ("Yes", "No"))

# Convert inputs to numeric values for model
def encode(value):
    return 1 if value == "Yes" else 0

inputs = np.array([[age, encode(smoking), encode(anxiety), encode(chronic_disease), encode(fatigue)]])

# Prediction button
if st.button("Predict"):
    prediction = model.predict(inputs)
    risk = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.success(f"Lung Cancer Prediction: {risk}")