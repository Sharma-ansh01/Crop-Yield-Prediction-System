import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import pickle
from sklearn.preprocessing import OrdinalEncoder

# Load data
data = pd.read_csv(r"C:\Users\anshh\OneDrive\Desktop\Crop_Yield-master\Crop_Yield-master\crop_production.csv")
with open(r"encoder.pkl", 'rb') as file:
    encoder = pickle.load(file)
with open(r"scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

# Set target and remove from data
# y = data['Production']
# X = data.drop(columns='Production')

# Data Splitting
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Load your trained models
# with open(r'C:\Users\anshh\OneDrive\Desktop\Crop_Yield-master\Crop_Yield-master\regressor.pkl', 'rb') as file:
#     clf = pickle.load(file)
with open(r"C:\Users\anshh\OneDrive\Desktop\Crop_Yield-master\XGBregressor2.pkl", 'rb') as file:
    xgb_model = pickle.load(file)


# Prepare for prediction
def prepare_input(state, district, season, crop, area, crop_year=2000):
    input_data = pd.DataFrame({
        'State_Name': [state],
        'District_Name': [district],
        'Crop_Year' : [crop_year],
        'Season': [season],
        'Crop': [crop],
        'Area': [area]
    })
    
    # encoder = OrdinalEncoder()
    input_data[['State_Name', 'District_Name', 'Season', 'Crop']] = encoder.transform(input_data[['State_Name', 'District_Name', 'Season', 'Crop']])
    print(input_data.shape)
    scaled_data = scaler.transform(input_data)
    print(scaled_data)
    
    # input_encoded = pd.get_dummies(input_data)
    # input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
    return scaled_data
    
st.title('Crop Yield Prediction')
data["Season"] = data["Season"].apply(lambda x: x.replace(" ", ""))
state = st.selectbox('State:', data['State_Name'].unique())
district = st.selectbox('District:', data['District_Name'].unique())
season = st.selectbox('Season:', data['Season'].unique())
crop = st.selectbox('Crop:', data['Crop'].unique())
area = st.number_input('Area:', min_value=0.0, step=0.1)

if st.button('Predict'):
    input_data = prepare_input(state, district, season, crop, area)
    
    # Make predictions using Random Forest and XGBoost models
    # rf_prediction = clf.predict(input_data)
    xgb_prediction = xgb_model.predict(input_data)
    
    # Display the predictions
    #st.write(f'Random Forest Prediction: {rf_prediction[0]}')
    st.write(f'XGBoost Prediction: {xgb_prediction[0]}')
    st.write(f'Your yield will be: {round(xgb_prediction[0], 2)} (approximately)')
