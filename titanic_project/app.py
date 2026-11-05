import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")

st.title("Titanic Survival Predictor")
st.write("Machine Learning Prediction App")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses", 0, 8, 0)
parch = st.number_input("Parents/Children", 0, 6, 0)
fare = st.number_input("Ticket Fare", 0.0, 600.0, 30.0)

if st.button("Predict"):
    gender = 0 if sex == "Male" else 1
    data = pd.DataFrame([[pclass, gender, age, sibsp, parch, fare]],
        columns=["pclass", "sex", "age", "sibsp", "parch", "fare"])
    result = model.predict(data)[0]

    if result == 1:
        st.success("Prediction: Survived")
    else:
        st.error("Prediction: Not Survived")
