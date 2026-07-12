#==================
# app.py
#==================

import streamlit as st
import pandas as pd
import pickle
import os

# Imports for EDA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# --- EDA Section ---
st.subheader("Exploratory Data Analysis (Optional)")
with st.expander("View Data Overview and Distributions"):
    st.write("Here are some key visualizations from the initial data exploration of `data.csv`:")

    # Load the original data for EDA
    @st.cache_data
    def load_eda_data():
        try:
            return pd.read_csv("data.csv")
        except FileNotFoundError:
            st.error("Error: 'data.csv' not found for EDA. Please ensure it's in the same directory for visualizations.")
            st.stop()

    eda_df = load_eda_data()

    if eda_df is not None:
        # Class Distribution
        st.write("#### Class Distribution")
        TARGET_COLUMN_EDA = " Label" # Ensure this matches the column name in data.csv
        if TARGET_COLUMN_EDA in eda_df.columns:
            fig_class_dist = plt.figure(figsize=(8, 5))
            sns.countplot(x=eda_df[TARGET_COLUMN_EDA], order=eda_df[TARGET_COLUMN_EDA].value_counts().index)
            plt.xticks(rotation=45)
            plt.title("Class Distribution")
            plt.tight_layout()
            st.pyplot(fig_class_dist)
            plt.close(fig_class_dist) # Close figure to free memory
        else:
            st.warning(f"Target column '{TARGET_COLUMN_EDA}' not found in data.csv for class distribution.")

        # Correlation Heatmap
        st.write("#### Feature Correlation Heatmap (Numeric Features)")
        numeric_columns_eda = eda_df.select_dtypes(include=np.number).columns
        if not numeric_columns_eda.empty:
            corr_eda = eda_df[numeric_columns_eda].corr()

            fig_heatmap = plt.figure(figsize=(15, 12))
            sns.heatmap(corr_eda, cmap="coolwarm", center=0, annot=False, fmt=".2f")
            plt.title("Feature Correlation Heatmap")
            st.pyplot(fig_heatmap)
            plt.close(fig_heatmap)
        else:
            st.info("No numeric columns found for correlation heatmap.")

        # Example Feature Histograms (First 5 Numeric Features)
        st.write("#### Example Feature Histograms (First 5 Numeric Features)")
        example_features = numeric_columns_eda[:5]
        if not example_features.empty:
            for feature in example_features:
                fig_hist = plt.figure(figsize=(7, 4))
                sns.histplot(eda_df[feature], kde=True)
                plt.title(f"Histogram: {feature}")
                st.pyplot(fig_hist)
                plt.close(fig_hist)
        else:
            st.info("No numeric columns found to plot example histograms.")
