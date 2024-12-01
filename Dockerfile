# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip and install required Python packages
RUN pip install -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Run Streamlit app with CORS disabled
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"]
