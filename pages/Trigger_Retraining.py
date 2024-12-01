import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model


def retrain_model(uploaded_file):
    # Load the existing model
    model = load_model("model1_simple.h5")

    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(df.head())

    # Define new encoders and scaler
    new_label_encoders = {}
    new_scaler = StandardScaler()

    # Encode categorical columns using new encoders
    for column in df.select_dtypes(include=['object']).columns:
        # Fit new label encoder to each categorical column
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str))
        new_label_encoders[column] = encoder
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

    # Scale the features using the new scaler
    X_scaled = new_scaler.fit_transform(X)

    # Retrain the model
    model.fit(X_scaled, y, epochs=10, batch_size=32)

    # Save updated model
    model.save("retrained_model1_simple.h5")

    # Save the new label encoders and scaler
    joblib.dump(new_scaler, "new_scaler.pkl")
    joblib.dump(new_label_encoders, "new_label_encoders.pkl")

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
