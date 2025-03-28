import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
model = pickle.load(open("logistic_regression_model.pkl", "rb"))

# Streamlit UI
st.title("Titanic Survival Prediction 🚢")

st.markdown(
    """
    Enter passenger details to predict if they would survive the Titanic disaster.
    """
)

# User input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=50.0)
sex = st.radio("Sex", ["Male", "Female"])
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"], index=2)

# Convert categorical inputs to numerical
sex_female = 1 if sex == "Female" else 0
sex_male = 1 if sex == "Male" else 0
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Prepare input data
input_data = np.array([[pclass, age, sibsp, parch, fare, sex_female, sex_male, embarked_C, embarked_Q, embarked_S]])
columns = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_female", "Sex_male", "Embarked_C", "Embarked_Q", "Embarked_S"]
input_df = pd.DataFrame(input_data, columns=columns)

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"🟢 The passenger **survives**! (Survival Probability: {prediction_proba:.2f})")
    else:
        st.error(f"🔴 The passenger **does not survive**. (Survival Probability: {prediction_proba:.2f})")
