import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load pre-trained model and preprocessing objects
model = tf.keras.models.load_model("model1_simple.h5")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")


def safe_transform(encoder, value, default="New to model"):
    """Handle unseen categories by assigning a default label."""
    try:
        return encoder.transform([value])[0]
    except ValueError:
        # If the value is unseen, return a default value
        return encoder.transform([default])[0]


def main():
    st.title("Model Prediction")
    st.write("Use the model to predict wildlife categories based on input data.")

    # Single prediction
    st.subheader("Single Prediction")
    species_name = st.text_input("Species Name (e.g., Panthera tigris):")
    systems = st.selectbox("System Type:", label_encoders["systems"].classes_)
    scopes = st.selectbox("Scope Type:", label_encoders["scopes"].classes_)

    if st.button("Predict"):
        # Preprocess inputs
        try:
            input_data = [[
                safe_transform(label_encoders["speciesName"], species_name),
                safe_transform(label_encoders["systems"], systems),
                safe_transform(label_encoders["scopes"], scopes)
            ]]
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)
            predicted_category_index = prediction.argmax()

            # Decode the predicted category (use the Category encoder)
            predicted_category = label_encoders["Category"].inverse_transform(
                [predicted_category_index])[0]

            # If the result is "New to model", display it as a positive result
            if predicted_category == "New to model":
                st.success(
                    f"Predicted Category: {predicted_category}", icon="✅")
            else:
                st.success(f"Predicted Category: {predicted_category}")
        except KeyError as e:
            st.error(f"Error: {e}. Ensure that the inputs are valid.")

    # Bulk prediction
    st.subheader("Bulk Prediction")
    uploaded_file = st.file_uploader("Upload a CSV File", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())

        # Check for necessary columns
        required_columns = ['speciesName', 'systems', 'scopes']
        if all(col in df.columns for col in required_columns):
            if st.button("Run Bulk Predictions"):
                # Encode categorical columns for the bulk data
                for col in required_columns:
                    df[col] = df[col].apply(
                        lambda x: safe_transform(label_encoders[col], x))

                # Scaling the features
                scaled_data = scaler.transform(df[required_columns])
                predictions = model.predict(scaled_data)
                predicted_categories = np.argmax(predictions, axis=1)

                # Decode predicted categories
                df["Predicted Category"] = label_encoders["Category"].inverse_transform(
                    predicted_categories)

                # Ensure "New to model" is handled correctly in the bulk predictions
                df["Predicted Category"] = df["Predicted Category"].apply(
                    lambda x: x if x != "New to model" else "New to model")

                # Display results
                st.write("Predictions:")
                st.dataframe(df)
                st.download_button("Download Predictions", df.to_csv(
                    index=False), "predictions.csv")
        else:
            st.error(
                f"Uploaded CSV must contain the following columns: {required_columns}")


if __name__ == "__main__":
    main()
