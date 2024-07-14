## Streamlit app to Get the input data and predict the customer whether Churn or not
## import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

## Load the Models
## Load the trainer models
model = tf.keras.models.load_model("pickle/model.h5")

## Load the Encoder
with open("pickle/lable_encoder_gen.pkl",'rb') as file:
    lable_encoder_gender=pickle.load(file)

with open("pickle/onehot_encoder_geo.pkl","rb") as file:
    onehot_encoder_geo=pickle.load(file)

with open("pickle/scaler.pkl","rb") as file:
    scaler=pickle.load(file)

## Get the input values
## Streamlit app
## userinput

st.title("Customer Churn Prediction")

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', lable_encoder_gender.classes_)
age = st.slider('Age',15,100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,10)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active member',[0,1])

## Convert input data into Dataframe

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [lable_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## Convert the test into numbers

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

prediction=model.predict(input_data_scaled)

prediction_proba = prediction[0][0]
if prediction[0][0] > 0.5:
    st.write("The customer is likely Churn")
else:
    st.write("The Custoer is likely not Chrun")

st.write(f"The probability is {prediction[0][0]}")

