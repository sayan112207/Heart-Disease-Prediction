import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Heart Disease Predictor", page_icon=":heart:")

# Loading the dataset
heartdata = pd.read_csv('heart_disease_data.csv')

# Splitting the dataset into features and target variable
x = heartdata.drop(columns=['target'], axis=1)
y = heartdata['target']

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Training the logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Saving the trained model using joblib and pickle
joblib.dump(model, 'heart_disease_predictor.joblib')
filename = 'heart_disease_predictor.pkl'
pickle.dump(model, open(filename, 'wb'))

# Loading the trained model
loaded_model = pickle.load(open('heart_disease_predictor.pkl', 'rb'))

# Defining the prediction function
def heart_disease_prediction(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]

# Creating the streamlit app
st.title("Heart Disease Predictor")

age = st.slider("Age", 0, 120, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 0, 200, 120)
chol = st.slider("Serum Cholesterol (mg/dL)", 0, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
thalach = st.slider("Maximum Heart Rate Achieved", 0, 250, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 7.0, 3.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Flourosopy", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

if st.button("Predict"):
    sex = 0 if sex == "Male" else 1
    result = heart_disease_prediction(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    if result == 0:
        st.success("You have a low risk of heart disease.")
    else:
        st.error("You have a high risk of heart disease.")