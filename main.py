import streamlit as st
import numpy as np 
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder,StandardScaler


# load the prediction object 
label_encoder = joblib.load(filename="label_encoder.pkl")
scaler =  joblib.load(filename="scaler.pkl")
model = joblib.load(filename="model_v1.pkl")

st.title("TIPS PREDICTION SYSTEM")
column1, column2 = st.column():
    

with column1:
    total_bill = st. number_input(label = 'Total Bill')
    sex = st.selectbox(label = "Gender", options = ['Male','Female'])
    smoker = st.selectbox(label = "Smoker", options = ['Yes','No'])
with column2:
    day = st.selectbox(label = "Day", options = list[main_data['Day'].unique)
    time = st.selectbox(label = "Time", options = ['Male','Female'])
    size = st.selectbox(label = "Gender", options = ['Male','Female'])

if st.button(label='predict'):
    st.divider 
    sample_dict = {
        'total_bill':[total_bill],  'sex':[sex],    'smoker':[smoker],
        'day':[day],    'time':[time],    'size':[size]
    data = pd.DataFrame(sample_dict)
    
    # encode the categorical column
    for col in cat_cols:
        encoder = label_encoders(col)
        data[col] = encoder.transform(data[col])
        
    # scale the dataset 
    data = scaler.transform(data)
    # get model prediction
    prediction = model.predict(data)
    st.success(f'This user is expected to tip around ${round(prediction)2,1}')
    
    }