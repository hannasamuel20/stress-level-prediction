import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('../model/gradientBoostingClassifier.pkl')


# App title
st.title("Stress Level Prediction")
st.subheader("Sample Data")
st.write(
    pd.DataFrame(
        {
            "Country": ["Australia"],
            "Year": [2008],
            "Status": ["Developed"],
            "Life expectancy": [81.3],
            "Adult Mortality": [66.0],
            "Alcohol": [10.76],
            "percentage expenditure": [8547.292357],
            "Hepatitis B": [94.0],
            "Measles": [65],
            "BMI": [62.9],
            "under-five deaths": [1],
            "Polio": [92.0],
            "Total expenditure": [8.78],
            "Diphtheria": [92.0],
            "HIV/AIDS": [0.1],
            "GDP": [49664.6854],
            "Population": [212492.0],
            "thinness 1-19 years": [0.7],
            "Income composition of resources": [0.921],
            "Schooling": [19.1],
        }
    )
)

# # Input fields
st.write("Enter features:")


study_hours_per_day = st.slider(label="Study Hours Per Day", min_value=0.0,
                                max_value=24.0,
                                value=5.6, step=0.1)

extracurricular_hours_per_day = st.slider(label="Extracurricular Hours Per Day", min_value=0.0,
                                          max_value=24.0,
                                          value=7.6, step=0.1)

sleep_hours_per_day = st.slider(label="Sleep Hours Per Day", min_value=0.0,
                                max_value=24.0,
                                value=3.6, step=0.1)

physical_activities_hours_per_day = st.slider(label="Physical Activities Hours Per Day", min_value=0.0,
                                              max_value=24.0,
                                              value=7.6, step=0.1)
gpa = st.slider(label="GPA", min_value=0.0,
                max_value=4.0,
                value=3.6, step=0.1)

if st.button("Predict"):

    input_data = np.array([study_hours_per_day, extracurricular_hours_per_day,
                                    sleep_hours_per_day, physical_activities_hours_per_day, gpa]).reshape(1, -1)
    prediction = model.predict(input_data)
    st.write(f"Prediction: {prediction[0]}")
