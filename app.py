
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder # Ensure these are imported as they might be used by the model

st.set_page_config(page_title="Network Intrusion Detection", layout="wide")

# Load the trained model, label encoder, and feature list
# @st.cache_resource is used for caching resources like models that are expensive to load
@st.cache_resource
def load_artifacts():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        feature_names = pickle.load(open('features.pkl', 'rb'))
        return model, label_encoder, feature_names
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please ensure 'model.pkl', 'label_encoder.pkl', and 'features.pkl' are in the same directory.")
        st.stop()

model, label_encoder, feature_names = load_artifacts()

# Streamlit UI
st.title("🕵️‍♂️ Network Intrusion Detection System")
st.markdown("---")

st.header("Input Network Traffic Features")
st.write("Please enter the values for the network traffic features below to detect potential intrusions.")

# Create input fields dynamically based on feature_names
input_data = {}
num_cols = 3 # Number of columns for input fields
cols = st.columns(num_cols)

# Create a dictionary to hold default values or sample from X_train if available
default_values = {}
# You might want to get typical values from your training data, e.g., X_train.mean()
# For now, let's use a simple placeholder or zero
for feature in feature_names:
    default_values[feature] = 0.0 # Placeholder, replace with actual typical values

for i, feature in enumerate(feature_names):
    with cols[i % num_cols]:
        # Use a consistent key for each input widget
        input_data[feature] = st.number_input(f"{feature}", value=float(default_values.get(feature, 0.0)), key=f"input_{feature}")

st.markdown("---")

if st.button("Predict Intrusion"): 
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocessing (similar to how X_train was processed)
    # Ensure column order matches the training data
    input_df = input_df[feature_names] 
    
    # Handle infinite values (as done during training)
    input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    input_df.fillna(input_df.mean(), inplace=True) # Fill NaNs with mean of the input data for prediction

    # Feature Scaling: This is crucial if your model used StandardScaler
    # During training, `X_train_scaled` was created using a StandardScaler.
    # For deployment, you would typically save the fitted StandardScaler object (schalar) 
    # and load it here to transform new input data. 
    # If the chosen best_model was one that required scaling (Logistic Regression, KNN, SVC, MLPClassifier),
    # the incoming data for prediction must also be scaled using the *same fitted scaler*.
    # Since the `schalar` object wasn't explicitly saved, we need to make an assumption or warn.
    
    # Check if the loaded model requires scaling based on model's name (heuristic) or a flag
    model_requires_scaling = False
    if hasattr(model, 'feature_names_in_') and len(model.feature_names_in_) > 0:
        # A more robust check if the model has a `feature_names_in_` attribute and it's not empty.
        # This would imply a scikit-learn model which might have been trained on scaled data.
        # However, `StandardScaler` itself is also a model-like object that would need to be saved.
        pass
    
    # For this example, let's assume `Extra Trees` was the best model (which does NOT require scaling).
    # If the actual best model requires scaling, you would need to modify this part
    # to load a *pre-fitted* StandardScaler and apply it.
    
    # As `Extra Trees` does not use scaled data, we directly use `input_df`
    # If your best model was, e.g., Logistic Regression, you would need to have saved 'schalar' from Module 4.
    # For demonstration, we'll proceed without scaling, assuming the loaded model doesn't need it.
    # If your model needs scaling, uncomment and adapt the following:
    # try:
    #     # Load the pre-fitted scaler. This would need to be saved in Module 6 as well.
    #     scaler = pickle.load(open('scaler.pkl', 'rb')) 
    #     processed_input = scaler.transform(input_df)
    # except FileNotFoundError:
    #     st.warning("Scaler not found. Proceeding without scaling. Ensure your model can handle unscaled data.")
    #     processed_input = input_df
    # except Exception as e:
    #     st.error(f"Error during scaling: {e}. Check if the scaler matches input features.")
    #     processed_input = input_df

    processed_input = input_df # Assuming best model (Extra Trees) doesn't need explicit scaling here.
    
    # Make prediction
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    # Display result
    st.subheader("Prediction Results")
    predicted_class_index = prediction[0]
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]

    st.write(f"The predicted network traffic type is: **{predicted_class_label}**")
    
    # Display probabilities for each class
    st.write("Confidence Scores:")
    proba_df = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
    st.dataframe(proba_df.style.format("{:.2%}"))

    if predicted_class_label == label_encoder.inverse_transform([0])[0]: # Assuming 0 is the benign class
        st.success("The network traffic appears to be **BENIGN** (Normal).")
    else:
        st.warning("Potential **INTRUSION DETECTED**! Please investigate further.")

st.markdown("--- ")
st.caption("This is a demonstration of a Machine Learning model deployed with Streamlit.")
