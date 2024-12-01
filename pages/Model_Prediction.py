import joblib
import numpy as np
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
    systems = st.selectbox("System Type:", label_encoders["systems"].classes_)
    scopes = st.selectbox("Scope Type:", label_encoders["scopes"].classes_)

    if st.button("Predict"):
        try:
            # Check if the input values exist in label encoders
            if species_name not in label_encoders["speciesName"].classes_:
                st.warning(
                    f"Species '{species_name}' is not recognized. Returning 'New to model'.")
                st.success("Predicted Category: New to model")
                return
            if systems not in label_encoders["systems"].classes_:
                st.warning(
                    f"System '{systems}' is not recognized. Returning 'New to model'.")
                st.success("Predicted Category: New to model")
                return
            if scopes not in label_encoders["scopes"].classes_:
                st.warning(
                    f"Scope '{scopes}' is not recognized. Returning 'New to model'.")
                st.success("Predicted Category: New to model")
                return

            # Encode and scale the input data
            input_data = [[
                label_encoders["speciesName"].transform([species_name])[0],
                label_encoders["systems"].transform([systems])[0],
                label_encoders["scopes"].transform([scopes])[0]
            ]]
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)
            predicted_category_index = prediction.argmax()

            # Decode the predicted category (use the Category encoder)
            predicted_category = label_encoders["Category"].inverse_transform(
                [predicted_category_index])[0]

            st.success(f"Predicted Category: {predicted_category}")

        except Exception as e:
            st.warning(f"Prediction failed: {e}. Returning 'New to model'.")
            st.success("Predicted Category: New to model")

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
                # Apply encoding and scaling for each row
                for idx, row in df.iterrows():
                    try:
                        # Check for unseen labels and display a warning if any
                        if row["speciesName"] not in label_encoders["speciesName"].classes_:
                            df.at[idx, "Predicted Category"] = "New to model"
                            continue
                        if row["systems"] not in label_encoders["systems"].classes_:
                            df.at[idx, "Predicted Category"] = "New to model"
                            continue
                        if row["scopes"] not in label_encoders["scopes"].classes_:
                            df.at[idx, "Predicted Category"] = "New to model"
                            continue

                        # Encode and scale the input data
                        input_data = [[
                            label_encoders["speciesName"].transform(
                                [row["speciesName"]])[0],
                            label_encoders["systems"].transform(
                                [row["systems"]])[0],
                            label_encoders["scopes"].transform(
                                [row["scopes"]])[0]
                        ]]
                        scaled_data = scaler.transform(input_data)
                        prediction = model.predict(scaled_data)
                        predicted_category_index = prediction.argmax()

                        # Decode the predicted category
                        predicted_category = label_encoders["Category"].inverse_transform(
                            [predicted_category_index])[0]
                        df.at[idx, "Predicted Category"] = predicted_category

                    except Exception as e:
                        # Handle errors by assigning "New to model"
                        df.at[idx, "Predicted Category"] = "New to model"
                        continue

                st.write("Predictions:")
                st.dataframe(df)
                st.download_button("Download Predictions", df.to_csv(
                    index=False), "predictions.csv")
        else:
            st.error(
                f"Uploaded CSV must contain the following columns: {required_columns}")


if __name__ == "__main__":
    main()
