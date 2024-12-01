import streamlit as st

# Import pages
from pages import (Home, Data_Visualizations, Model_Prediction,
                   Trigger_Retraining, About_and_Documentation)

# Define pages
PAGES = {
    "Home": Home.main,
    "Data Visualizations": Data_Visualizations.main,
    "Model Prediction": Model_Prediction.main,
    "Trigger Retraining": Trigger_Retraining.main,
    "About & Documentation": About_and_Documentation.main,
}

# Dropdown for navigation
selection = st.selectbox("Select a page", list(PAGES.keys()))

# Render the selected page
PAGES[selection]()
