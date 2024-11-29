import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf

# Load pre-trained model and preprocessing objects
model = tf.keras.models.load_model("model1_simple.h5")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")


def main():
    st.title("Model Prediction")
    st.write("Use the model to predict wildlife categories based on input data.")

    # Single prediction
    st.subheader("Single Prediction")
    species_name = st.text_input("Species Name (e.g., Panthera tigris):")
    systems = st.selectbox(
        "System Type:", label_encoders["systems"].classes_)
    scopes = st.selectbox("Scope Type:", label_encoders["scopes"].classes_)

    if st.button("Predict"):
        # Preprocess inputs
        try:
            input_data = [[
                label_encoders["speciesName"].transform([species_name])[0],
                label_encoders["systems"].transform([systems])[0],
                label_encoders["scopes"].transform([scopes])[0]
            ]]
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)
            st.success(f"Predicted Category: {prediction.argmax()}")
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
                # Encode categorical columns
                for col in required_columns:
                    df[col] = label_encoders[col].transform(df[col])
                scaled_data = scaler.transform(df[required_columns])
                predictions = model.predict(scaled_data)
                df["Predicted Category"] = predictions.argmax(axis=1)
                st.write("Predictions:")
                st.dataframe(df)
                st.download_button("Download Predictions", df.to_csv(
                    index=False), "predictions.csv")
        else:
            st.error(f"Uploaded CSV must contain the following columns: {required_columns}")


if __name__ == "__main__":
    main()
