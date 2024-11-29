import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


def main():
    st.title("Data Visualizations")
    st.write("Explore and visualize your data.")

    uploaded_file = st.file_uploader("Upload a Dataset", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Data:")
        st.dataframe(df.head())

        # Feature selection
        feature = st.selectbox("Select a feature to visualize:", df.columns)
        if feature:
            st.subheader(f"Visualization of {feature}")
            fig, ax = plt.subplots()
            sns.histplot(df[feature], kde=True, ax=ax)
            st.pyplot(fig)

        # Correlation heatmap
        if st.checkbox("Show Correlation Heatmap"):
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)


if __name__ == "__main__":
    main()
