
import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(layout="wide")

st.title("Network Intrusion Detection")
st.write("This application predicts network intrusion based on flow features.")

# Load the model, label encoder, and features
@st.cache_resource
def load_artifacts():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        features = pickle.load(open('features.pkl', 'rb'))
        return model, label_encoder, features
    except FileNotFoundError as e:
        st.error(f"Error loading necessary files: {e}. Make sure 'model.pkl', 'label_encoder.pkl', and 'features.pkl' are in the same directory.")
        st.stop()

model, label_encoder, features = load_artifacts()

st.sidebar.header("Input Features")

# Create input fields for each feature
input_data = {}
for feature in features:
    # Attempt to infer type for better input widgets
    if 'Flow Duration' in feature or 'Packet Length' in feature or 'IAT' in feature or 'Bytes/s' in feature:
        input_data[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0)
    elif 'Port' in feature or 'Packets' in feature or 'seg_size' in feature:
        input_data[feature] = st.sidebar.number_input(f"Enter {feature}", value=0, format="%d")
    elif 'Std' in feature or 'Mean' in feature or 'Min' in feature or 'Max' in feature:
        input_data[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0)
    elif 'Flag' in feature:
        input_data[feature] = st.sidebar.selectbox(f"Select {feature}", [0, 1])
    else:
        input_data[feature] = st.sidebar.text_input(f"Enter {feature}", value="0") # Default to text for unknown types

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess input data (e.g., scaling if the model was trained on scaled data)
# NOTE: If your best model (Extra Trees) was trained on unscaled data, you don't need scaling here.
# If you switch to a scaled model (e.g., Logistic Regression), you would need to implement StandardScaler here.

if st.button("Predict"):    
    try:
        prediction_encoded = model.predict(input_df)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)
        
        st.success(f"The predicted class is: **{prediction_label[0]}**")
        
        st.subheader("Raw Prediction Output")
        st.write(prediction_encoded)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check your input values and ensure they are numerical where expected.")

