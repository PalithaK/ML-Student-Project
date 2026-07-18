
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(layout='wide')

st.title('Network Intrusion Detection System')
st.write('This application uses a pre-trained machine learning model to detect network intrusions.')

# --- Load Model Artifacts ---

@st.cache_resource
def load_model_artifacts():
    model_path = 'model.pkl'
    label_encoder_path = 'label_encoder.pkl'
    features_path = 'features.pkl'
    scaler_path = 'scaler.pkl'

    if not os.path.exists(model_path):
        st.error(f'Model file not found: {model_path}. Please ensure it is in the same directory.')
        st.stop()
    if not os.path.exists(label_encoder_path):
        st.error(f'Label encoder file not found: {label_encoder_path}. Please ensure it is in the same directory.')
        st.stop()
    if not os.path.exists(features_path):
        st.error(f'Features file not found: {features_path}. Please ensure it is in the same directory.')
        st.stop()

    try:
        model = pickle.load(open(model_path, 'rb'))
        label_encoder = pickle.load(open(label_encoder_path, 'rb'))
        features = pickle.load(open(features_path, 'rb'))
        
        scaler = None
        if os.path.exists(scaler_path):
            scaler = pickle.load(open(scaler_path, 'rb'))
        else:
            st.warning('Scaler file not found (scaler.pkl). Assuming the model does not require scaling or was trained without it.')

        return model, label_encoder, features, scaler
    except Exception as e:
        st.error(f'Error loading model artifacts: {e}')
        st.stop()

model, label_encoder, features, scaler = load_model_artifacts()

# Input Features (Placeholder)
st.header('Enter Network Traffic Features')
# st.write('For a real application, you would add input widgets here for each feature.')
st.write('---')


input_data = {}
# For simplicity, dummy values are created for the first 5 features

st.subheader('Example Input (replace with actual widgets):')
for i, feature_name in enumerate(features[:5]): # Only showing first 5 for brevity
    input_data[feature_name] = st.slider(f'Input for {feature_name}',
                                         min_value=0.0,
                                         max_value=100.0,
                                         value=50.0,
                                         key=f'slider_{i}')

# Add the rest of the features with default values (or based on your dataset's typical values)
for feature_name in features[5:]:
    input_data[feature_name] = 0.0 # Default value for other features

input_df = pd.DataFrame([input_data])

st.subheader('Input Data Preview:')
st.dataframe(input_df)

# Prediction 
if st.button('Predict Intrusion'):
    try:
        # Ensure the input DataFrame has columns in the same order as trained features
        input_df_ordered = input_df[features]

        # Apply scaler if it exists
        if scaler:
            scaled_input = scaler.transform(input_df_ordered)
            prediction_input = scaled_input
        else:
            prediction_input = input_df_ordered

        prediction_encoded = model.predict(prediction_input)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)

        st.success(f'Prediction: **{prediction_label[0]}**')

        # Optional: Display prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(prediction_input)[0]
            proba_df = pd.DataFrame({
                'Class': label_encoder.classes_,
                'Probability': probabilities
            }).sort_values(by='Probability', ascending=False)
            st.subheader('Prediction Probabilities:')
            st.dataframe(proba_df.style.format({'Probability': '{:.2%}'}))

    except Exception as e:
        st.error(f'An error occurred during prediction: {e}')

st.write('---')
st.info('COM 763: Palitha Kuruvita / 25026175')
