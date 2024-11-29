import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import joblib

def retrain_model(data):
    # Load existing model
    model = load_model("model1_simple.h5")
    scaler = joblib.load("scaler.pkl")

    # Preprocess and train new data
    X = data.drop("Category", axis=1)
    y = data["Category"]
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y, epochs=10, batch_size=32)

    # Save updated model
    model.save("model1_simple.h5")

def main():
    st.title("Upload & Retrain")
    st.write("Upload new data to retrain the model.")

    uploaded_file = st.file_uploader("Upload a CSV File", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())

        if st.button("Start Retraining"):
            retrain_model(df)
            st.success("Model retrained successfully!")

if __name__ == "__main__":
    main()
