import os

import streamlit as st


def main():
    # Set the app title
    st.title("Data Visualizations")

    # Print current working directory to ensure correct path
    st.write(f"Current working directory: {os.getcwd()}")

    # Specify the directory containing the images (absolute or relative)
    image_dir = os.path.join(os.getcwd(), "visualizations")

    # List of image file names (no need for "visualizations/" prefix, as the directory is handled)
    image_files = [
        "category_distribution.png",
        "correlation_heatmap.png",
        "top_species_names.png"
    ]

    # Display images
    st.subheader("Visualizations")
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)  # Correct path generation
        try:
            st.image(img_path, caption=img_file.split(
                '.')[0], use_column_width=True)
        except FileNotFoundError:
            st.error(f"Image {img_file} not found in {img_path}.")


if __name__ == "__main__":
    main()
