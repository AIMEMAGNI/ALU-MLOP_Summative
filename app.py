import streamlit as st

# Import pages
from pages import (About_and_Documentation, Data_Visualizations, Home,
                   Model_Prediction, Trigger_Retraining, Upload_and_Retrain)

# Define pages
PAGES = {
    "Home": Home.main,
    "Model Prediction": Model_Prediction.main,
    "Data Visualizations": Data_Visualizations.main,
    "Upload & Retrain": Upload_and_Retrain.main,
    "Trigger Retraining": Trigger_Retraining.main,
    "About & Documentation": About_and_Documentation.main,
}

# Dropdown for navigation
selection = st.selectbox("Select a page", list(PAGES.keys()))

# Render the selected page
PAGES[selection]()
