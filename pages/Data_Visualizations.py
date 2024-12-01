import streamlit as st


def main():
    # Set the app title
    st.title("Data Visualizations")

    # List of image URLs
    image_urls = [
        "https://github.com/AIMEMAGNI/ALU-MLOP_Summative/blob/main/pages/visualizations/category_distribution.png?raw=true",
        "https://github.com/AIMEMAGNI/ALU-MLOP_Summative/blob/main/pages/visualizations/correlation_heatmap.png?raw=true",
        "https://github.com/AIMEMAGNI/ALU-MLOP_Summative/blob/main/pages/visualizations/top_species_names.png?raw=true"
    ]

    # Display images
    st.subheader("Visualizations")
    for img_url in image_urls:
        st.image(img_url, caption=img_url.split('/')
                 [-1].split('.')[0], use_column_width=True)


if __name__ == "__main__":
    main()
