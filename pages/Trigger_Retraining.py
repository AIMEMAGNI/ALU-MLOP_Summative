import joblib
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# Load pre-trained model and tools
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")


def encode_data(df):
    """Encodes categorical columns using the pre-loaded label encoders."""
    for column, encoder in label_encoders.items():
        if column in df.columns:
            # Apply transformation and handle unknown values by assigning -1
            df[column] = df[column].apply(lambda x: encoder.transform([x])[
                                          0] if x in encoder.classes_ else -1)
    return df


def retrain_model(uploaded_df):
    # Load the existing model
    model = load_model("model1_simple.h5")

    # Encode categorical columns first
    uploaded_df = encode_data(uploaded_df)

    # Prepare data for training
    X = uploaded_df.drop("Category", axis=1)
    y = uploaded_df["Category"]

    # Check if all features are numeric after encoding (for scaling)
    if not all(X.dtypes.apply(lambda x: x in ['int64', 'float64'])):
        raise ValueError(
            "Some columns still contain non-numeric data after encoding.")

    # Scale the features
    X_scaled = scaler.fit_transform(X)

    # Retrain the model
    model.fit(X_scaled, y, epochs=10, batch_size=32)

    # Save updated model
    model.save("Models/retrained_model1_simple.h5")
    return "Retraining completed successfully!"


def main():
    st.title("Retrain Model")
    st.write("Trigger the retraining of the model with updated data.")

    uploaded_file = st.file_uploader(
        "Upload a new CSV dataset for retraining", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())

        if st.button("Start Retraining"):
            try:
                result = retrain_model(df)
                st.success(result)
            except Exception as e:
                st.error(f"Retraining failed: {e}")


if __name__ == "__main__":
    main()
