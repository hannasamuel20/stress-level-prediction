import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('./model/gradientBoostingClassifier.pkl')


# App title
st.title("Stress Level Prediction")
st.subheader("Sample Data")
st.write(
    pd.DataFrame(
        {
            "Study_Hours_Per_Day": [5.5],
            "Extracurricular_Hours_Per_Day": [1.8],
            "Sleep_Hours_Per_Day": [7.7],
            "Social_Hours_Per_Day": [1.5],
            "Physical_Activity_Hours_Per_Day": [7.5],
            "GPA": [3.05],
        }
    )
)

# # Input fields
st.write("Enter features:")


study_hours_per_day = st.slider(label="Study Hours Per Day", min_value=0.0,
                                max_value=24.0,
                                value=5.5, step=0.1)

extracurricular_hours_per_day = st.slider(label="Extracurricular Hours Per Day", min_value=0.0,
                                          max_value=24.0,
                                          value=1.8, step=0.1)

sleep_hours_per_day = st.slider(label="Sleep Hours Per Day", min_value=0.0,
                                max_value=24.0,
                                value=7.7, step=0.1)
social_hours_per_day = st.slider(label="Social Hours Per Day", min_value=0.0,
                                 max_value=24.0,
                                 value=1.5, step=0.1)

physical_activities_hours_per_day = st.slider(label="Physical Activities Hours Per Day", min_value=0.0,
                                              max_value=24.0,
                                              value=7.5, step=0.1)
gpa = st.slider(label="GPA", min_value=0.0,
                max_value=4.0,
                value=3.05, step=0.1)

if st.button("Predict"):
    input_data = np.array([study_hours_per_day, extracurricular_hours_per_day,
                           sleep_hours_per_day, social_hours_per_day, physical_activities_hours_per_day, gpa]).reshape(1, -1)
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        st.write("Prediction: Low")
    elif prediction[0] == 1:
        st.write("Prediction: Moderate")
    else:
        st.write("Prediction: High")
