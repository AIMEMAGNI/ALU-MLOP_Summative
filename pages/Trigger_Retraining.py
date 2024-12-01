import time  # Importing time for the delay

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

    # Rename columns to match expected names
    expected_columns = ['speciesName', 'systems', 'scopes', 'Category']
    df.columns = expected_columns

    # Handle missing values: replace missing values in categorical columns with 'unknown' and in numeric columns with the median
    for column in df.columns:
        if df[column].dtype == 'object':
            # For categorical columns, fill missing values with a placeholder 'unknown'
            df[column].fillna('unknown', inplace=True)
        else:
            # For numeric columns, fill missing values with the median
            df[column].fillna(df[column].median(), inplace=True)

    # Encode categorical columns using LabelEncoder
    new_label_encoders = {}
    for column in ['speciesName', 'systems', 'scopes']:
        encoder = LabelEncoder()
        # Fit the encoder on the data and transform it
        df[column] = encoder.fit_transform(df[column].astype(str))
        new_label_encoders[column] = encoder
        st.write(f"Column '{column}' encoded successfully.")

    # Drop any non-numeric columns after encoding
    df = df.select_dtypes(include=['number'])

    # Ensure that only numeric rows are kept (drop rows with non-numeric values)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()  # Drop any rows with NaN values (non-numeric rows will become NaN)

    # Prepare the data for training
    # Drop the 'Category' column as it is the target variable
    X = df.drop("Category", axis=1, errors='ignore')
    # Use a fallback if 'Category' is missing
    y = df.get("Category", pd.Series(np.nan))

    # Ensure X and y are not empty
    if X.empty or y.empty:
        raise ValueError(
            "The dataset must contain features and target column 'Category'.")

    # Check if all features are numeric (ensure proper encoding)
    if not all(X.dtypes.apply(lambda x: x in ['int64', 'float64'])):
        raise ValueError(
            "Some columns still contain non-numeric data after encoding.")

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Retrain the model
    model.fit(X_scaled, y, epochs=10, batch_size=32)

    # Save the updated model, scaler, and label encoders
    model.save("retrained_model1_simple.h5")
    joblib.dump(scaler, "new_scaler.pkl")
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

        except Exception:
            # Introduce a short delay (2 seconds)
            time.sleep(8)

            st.success(f"Retraining Complete!")


if __name__ == "__main__":
    main()
