import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


def retrain_model(uploaded_file):
    # Load pre-trained model and tools
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")

    # Load the existing model
    model = load_model("model1_simple.h5")

    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(df.head())

    # Encode categorical columns using the pre-loaded label encoders
    for column, encoder in label_encoders.items():
        if column in df.columns:
            # Encode each column using its specific encoder
            def safe_encode(x):
                try:
                    return encoder.transform([x])[0]
                except ValueError:
                    return -1  # Handle unseen categories by assigning -1
            df[column] = df[column].apply(safe_encode)
            st.write(f"Column '{column}' encoded successfully.")

    # Prepare the data for training
    # Ignore errors if 'Category' column is missing
    X = df.drop("Category", axis=1, errors='ignore')
    # Use a fallback in case 'Category' column is missing
    y = df.get("Category", pd.Series(np.nan))

    # Ensure X and y are not empty
    if X.empty or y.empty:
        raise ValueError(
            "The dataset must contain features and target column 'Category'.")

    # Check if all features are numeric (ensure proper encoding)
    if not all(X.dtypes.apply(lambda x: x in ['int64', 'float64'])):
        raise ValueError(
            "Some columns still contain non-numeric data after encoding.")

    # Scale the features
    try:
        X_scaled = scaler.fit_transform(X)
    except ValueError as e:
        raise ValueError(f"Scaling failed: {e}")

    # Retrain the model
    model.fit(X_scaled, y, epochs=10, batch_size=32)

    # Save updated model
    model.save("retrained_model1_simple.h5")

    return "Retraining completed successfully!"


def main():
    st.title("Retrain Model")
    st.write("Trigger the retraining of the model with updated data.")

    uploaded_file = st.file_uploader(
        "Upload a new CSV dataset for retraining", type="csv")

    if uploaded_file:
        try:
            result = retrain_model(uploaded_file)
            st.success(result)
        except Exception as e:
            st.error(f"Retraining failed: {e}")


if __name__ == "__main__":
    main()
