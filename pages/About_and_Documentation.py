import streamlit as st


def main():
    st.title("About Wildlife Insight")
    st.markdown("""
    ### Wildlife Insight: AI-Powered Conservation  
    This app is designed to help in wildlife conservation efforts by leveraging machine learning models for classification, data analysis, and predictions. The application integrates cutting-edge AI techniques with an easy-to-use interface for conservationists and researchers.

    #### Key Features:
    - **Model Prediction**: Upload data to predict wildlife categories.
    - **Data Visualizations**: Explore patterns and insights in data.
    - **Retrain the Model**: Keep the model updated with the latest data.
    - **Performance Testing**: Simulate load and evaluate model performance under stress.
    """)

    st.markdown("""
    ### Technical Documentation  
    1. **Data Input**:
       - Data should be in CSV format.
       - For single predictions, ensure features match those used in model training.
    2. **Model**:
       - The classification model is a TensorFlow neural network.
       - Pre-trained with scaled inputs and categorical encoding for features.
    3. **Retraining Process**:
       - Trigger retraining with new data via the Retrain page.
       - Ensure uploaded data has the same structure as the training data.
    4. **Performance Testing**:
       - Load tested using Locust (instructions available in the repository).
    """)

    st.markdown("""
    ### Contribute or Report Issues  
    If you encounter issues or have suggestions, feel free to contribute:
    - **GitHub Repository**: [Wildlife Insight Repo](https://github.com/AIMEMAGNI/ALU-MLOP_Summative)
    - **Support Email**: a.ndayishim@alustudent.com
    """)

    st.image("https://iucn-members.us/wp-content/uploads/2024/01/KenyaElephants-RickBerstrom-CCbyND.jpg",
             caption="AI for Conservation")


if __name__ == "__main__":
    main()
