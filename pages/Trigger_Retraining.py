import streamlit as st
from tensorflow.keras.models import load_model
import joblib
import pandas as pd

# Load pre-trained model and tools
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

def retrain_model(uploaded_df):
    # Load the existing model
    model = load_model("model1_simple.h5")
    
    # Prepare data for training
    X = uploaded_df.drop("Category", axis=1)
    y = uploaded_df["Category"]
    X_scaled = scaler.fit_transform(X)
    
    # Retrain the model
    model.fit(X_scaled, y, epochs=10, batch_size=32)
    
    # Save updated model
    model.save("model1_simple.h5")
    return "Retraining completed successfully!"

def main():
    st.title("Retrain Model")
    st.write("Trigger the retraining of the model with updated data.")

    uploaded_file = st.file_uploader("Upload a new CSV dataset for retraining", type="csv")

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
