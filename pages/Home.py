import streamlit as st


def main():
    st.title("Welcome to Wildlife Insight")
    st.write("""
        Wildlife Insight leverages machine learning to classify species and analyze data efficiently.  
        Use the sidebar to navigate through the app's features.
    """)

    st.image("https://iucn-members.us/wp-content/uploads/2024/01/KenyaElephants-RickBerstrom-CCbyND.jpg",
             caption="Wildlife Conservation with AI")
    st.markdown("### Features:")
    st.markdown("- **Model Prediction**: Make predictions on species data.")
    st.markdown("- **Data Visualizations**: Explore and interpret data.")
    st.markdown("- **Upload & Retrain**: Update the model with new data.")
    st.markdown(
        "- **Performance Testing**: Test the model's scalability and response time.")


if __name__ == "__main__":
    main()
