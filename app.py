import streamlit as st
import pickle
import json
import os

# Load models and columns using relative paths (ensure files are in the repo)
with open("diabetes_model.pickle", "rb") as f:
    diabetes_model = pickle.load(f)
with open("heart_model.pickle", "rb") as f:
    heart_model = pickle.load(f)
with open("parkinsons_model.pkl", "rb") as f:
    parkinsons_model = pickle.load(f)

with open("diabetes_columns.json", "r") as f:
    diabetes_columns = json.load(f)
with open("heart_columns.json", "r") as f:
    heart_columns = json.load(f)
with open("parkinsons_columns.json", "r") as f:
    parkinsons_columns = json.load(f)

# Sidebar navigation
st.sidebar.title("Multiple Disease Prediction System")
selected_page = st.sidebar.radio(
    "Choose Prediction",
    ["Diabetes", "Heart Disease", "Parkinson's"]
)

def predict_diabetes():
    st.header("Diabetes Prediction using ML")
    inputs = {}
    for col in diabetes_columns:
        val = st.number_input(col.replace('_', ' ').title(), key=col)
        inputs[col] = val
    if st.button("Diabetes Test Result"):
        features = [inputs[col] for col in diabetes_columns]
        prediction = diabetes_model.predict([features])
        result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes"
        st.success(result)

def predict_heart():
    st.header("Heart Disease Prediction using ML")
    inputs = {}
    for col in heart_columns:
        val = st.number_input(col.replace('_', ' ').title(), key=col)
        inputs[col] = val
    if st.button("Heart Test Result"):
        features = [inputs[col] for col in heart_columns]
        prediction = heart_model.predict([features])
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
        st.success(result)

def predict_parkinsons():
    st.header("Parkinson's Disease Prediction using ML")
    inputs = {}
    for col in parkinsons_columns:
        val = st.number_input(col.replace('_', ' ').title(), key=col)
        inputs[col] = val
    if st.button("Parkinson's Test Result"):
        features = [inputs[col] for col in parkinsons_columns]
        prediction = parkinsons_model.predict([features])
        result = "Parkinson's Detected" if prediction[0] == 1 else "No Parkinson's Disease"
        st.success(result)

if selected_page == "Diabetes":
    predict_diabetes()
elif selected_page == "Heart Disease":
    predict_heart()
elif selected_page == "Parkinson's":
    predict_parkinsons()

# Optional: Custom CSS styling for Streamlit widgets
with open("app_style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
