import streamlit as st
import pickle
import json

# --- Load Models and Columns ---
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

# --- Custom CSS for Styles ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css('app_style.css')

# --- Sidebar Navigation ---
st.sidebar.title("Multiple Disease Prediction System")
page = st.sidebar.radio("Choose Prediction", ["Diabetes", "Heart Disease", "Parkinson's"])

# --- Main Page UI with Custom Form Layouts ---
def diabetes_form():
    st.markdown("<h1 style='margin-top:32px;'>Diabetes Prediction using ML</h1>", unsafe_allow_html=True)
    # Arrange fields in two rows for visual similarity
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)
    user_input = {}
    user_input[diabetes_columns[0]] = col1.number_input("Number of Pregnancies", min_value=0.0, key="pregnancies")
    user_input[diabetes_columns[1]] = col2.number_input("Glucose Level", min_value=0.0, key="glucose")
    user_input[diabetes_columns[2]] = col3.number_input("Blood Pressure", min_value=0.0, key="bloodpressure")
    user_input[diabetes_columns[3]] = col4.number_input("Skin Thickness Value", min_value=0.0, key="skinthickness")
    user_input[diabetes_columns[4]] = col5.number_input("Insulin Level", min_value=0.0, key="insulin")
    user_input[diabetes_columns[5]] = col6.number_input("BMI", min_value=0.0, key="bmi")
    user_input[diabetes_columns[6]] = col7.number_input("Diabetes Pedigree Function Value", min_value=0.0, key="diabetespedigreefunction")
    user_input[diabetes_columns[7]] = col8.number_input("Age of the person", min_value=0.0, key="age")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Diabetes Test Result", key="diabetes_submit"):
        features = [user_input[col] for col in diabetes_columns]
        pred = diabetes_model.predict([features])
        result = "Diabetes Detected" if pred[0] == 1 else "No Diabetes"
        st.success(result)

def heart_form():
    st.markdown("<h1 style='margin-top:32px;'>Heart Disease Prediction using ML</h1>", unsafe_allow_html=True)
    # Arrange all fields in rows of 4 columns each for clean layout
    cols = st.columns(4)
    user_input = {}
    for i, colname in enumerate(heart_columns):
        idx = i % 4
        row = i // 4
        if idx == 0 and i != 0:
            cols = st.columns(4)
        user_input[colname] = cols[idx].number_input(colname.replace("_", " ").title(), key=f"heart_{colname}")
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Heart Test Result", key="heart_submit"):
        features = [user_input[col] for col in heart_columns]
        pred = heart_model.predict([features])
        result = "Heart Disease Detected" if pred[0] == 1 else "No Heart Disease"
        st.success(result)

def parkinsons_form():
    st.markdown("<h1 style='margin-top:32px;'>Parkinson's Disease Prediction using ML</h1>", unsafe_allow_html=True)
    # Arrange fields in multiple rows of 4 columns (Parkinson's has many features)
    user_input = {}
    total = len(parkinsons_columns)
    for r in range(0, total, 4):
        cols = st.columns(4)
        for i in range(4):
            if r+i < total:
                colname = parkinsons_columns[r+i]
                user_input[colname] = cols[i].number_input(colname.replace("_", " ").title(), key=f"parkinsons_{colname}")
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Parkinson's Test Result", key="parkinsons_submit"):
        features = [user_input[col] for col in parkinsons_columns]
        pred = parkinsons_model.predict([features])
        result = "Parkinson's Detected" if pred[0] == 1 else "No Parkinson's Disease"
        st.success(result)

# -- Routing --
if page == "Diabetes":
    diabetes_form()
elif page == "Heart Disease":
    heart_form()
elif page == "Parkinson's":
    parkinsons_form()
