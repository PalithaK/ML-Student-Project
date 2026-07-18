
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Streamlit Page Configuration
st.set_page_config(page_title="Network Intrusion Detection System", layout="wide")

# --- Helper Functions ---
@st.cache_resource
def load_ml_artifacts():
    """Loads the trained model, label encoder, feature list, and scaler."""
    try:
        ml_model = pickle.load(open('model.pkl', 'rb'))
        target_label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        dataset_feature_names = pickle.load(open('features.pkl', 'rb'))
        
        data_scaler = None
        try:
            data_scaler = pickle.load(open('scaler.pkl', 'rb'))
        except FileNotFoundError:
            st.warning("Scaler file 'scaler.pkl' not found. Data will be processed without scaling.")

        return ml_model, target_label_encoder, dataset_feature_names, data_scaler
    except FileNotFoundError as e:
        st.error(f"Application startup failed: Missing ML artifact file - {e}. Ensure 'model.pkl', 'label_encoder.pkl', 'features.pkl' (and optionally 'scaler.pkl') are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading ML artifacts: {e}")
        st.stop()

# Load artifacts once
ml_model, target_label_encoder, dataset_feature_names, data_scaler = load_ml_artifacts()

# --- Streamlit UI Elements ---
st.title("🕵️‍♂️ Network Intrusion Detection System")
st.markdown("Welcome! This application predicts network traffic behavior based on various network features.")
st.markdown("---")

st.subheader("Enter Network Traffic Features")
st.write("Adjust the parameters below and click 'Analyze Traffic' to get a prediction.")

# Initialize a dictionary for input feature values
input_feature_values = {}

# Create input fields in a more organized layout
num_columns = 4 # Number of columns for input fields
columns = st.columns(num_columns)

# Pre-fill with example values or zeros/means for demonstration

example_values = {
    ' Destination Port': 80,
    ' Flow Duration': 3000000,
    ' Total Fwd Packets': 10,
    ' Total Backward Packets': 12,
    'Total Length of Fwd Packets': 100,
    ' Total Length of Bwd Packets': 200,
    ' Fwd Packet Length Max': 50,
    ' Fwd Packet Length Min': 10,
    ' Fwd Packet Length Mean': 30.0,
    ' Fwd Packet Length Std': 15.0,
    ' Bwd Packet Length Max': 70,
    ' Bwd Packet Length Min': 20,
    ' Bwd Packet Length Mean': 45.0,
    ' Bwd Packet Length Std': 20.0,
    'Flow Bytes/s': 100000.0,
    ' Flow Packets/s': 500.0,
    ' Flow IAT Mean': 20000.0,
    ' Flow IAT Std': 10000.0,
    ' Flow IAT Max': 500000,
    ' Flow IAT Min': 1000,
}

for i, feature_name in enumerate(dataset_feature_names):
    with columns[i % num_columns]:
        default_value = example_values.get(feature_name, 0.0) # Use 0.0 if no example value
        # Convert to float to ensure number_input works correctly for all numeric types
        input_feature_values[feature_name] = st.number_input(
            f"**{feature_name}**",
            value=float(default_value),
            format="%.4f", # Display with 4 decimal places for floats
            key=f"input_{feature_name}"
        )

st.markdown("--- ")

if st.button("⚡ Analyze Traffic"): 
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_feature_values])

    # Ensure column order matches the training data features
    input_df = input_df[dataset_feature_names]

    # Handle infinite and NaN values (consistent with training preprocessing)
    input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill any remaining NaNs with 0 (or a more sophisticated imputation strategy)
    input_df.fillna(0, inplace=True)

    # Apply scaling if a scaler was loaded and the model requires it
    processed_input_data = input_df
    if data_scaler is not None: # Check if scaler was loaded
        # List of models that use scaling (from Module 4)
        scaled_models = [
            "LogisticRegression",
            "KNeighborsClassifier",
            "SVC",
            "MLPClassifier"
        ]
        if any(model_type in str(type(ml_model)) for model_type in scaled_models):
            processed_input_data = data_scaler.transform(input_df)
            st.sidebar.info("Input data scaled for prediction.") 
        else:
            st.sidebar.info("Model does not require scaling. Proceeding with unscaled data.")

    try:
        # Make prediction
        prediction_result = ml_model.predict(processed_input_data)
        prediction_probabilities = ml_model.predict_proba(processed_input_data)

        # Display results
        st.subheader("Prediction Results")
        predicted_class_index = prediction_result[0]
        predicted_class_label = target_label_encoder.inverse_transform([predicted_class_index])[0]

        st.markdown(f"### Predicted Traffic Type: <span style='color: {'green' if predicted_class_label == 'BENIGN' else 'red'}; font-size: 28px;'>{predicted_class_label}</span>", unsafe_allow_html=True)

        if predicted_class_label == 'BENIGN':
            st.success("✅ The network traffic appears to be **BENIGN** (Normal).")
        else:
            st.error("🚨 Potential **INTRUSION DETECTED**! Immediate investigation is recommended.")

        st.markdown("#### Confidence Scores")
        probability_df = pd.DataFrame(prediction_probabilities, columns=target_label_encoder.classes_)
        st.dataframe(probability_df.style.format("{:.2%}").highlight_max(axis=1))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check the input values and ensure the model and data are compatible.")

st.markdown("--- ")
st.caption("COM 763 Assignment 01 : 25026175 / Palitha Kuruvita")

