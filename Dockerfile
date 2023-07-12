# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install necessary system packages and SUMO
RUN apt-get update && apt-get install -y \
    g++ \
    python3 \
    python3-dev \
    libxerces-c-dev \
    libfox-1.6-dev \
    libgdal-dev \
    libproj-dev \
    libgl2ps-dev \
    swig \
    sumo \
    sumo-tools \
    sumo-doc \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for SUMO
ENV SUMO_HOME=/usr/share/sumo
ENV PATH=$PATH:$SUMO_HOME/bin

# Install MLflow
RUN pip install mlflow

# Expose port for MLflow server
EXPOSE 5000

# Set the command to start MLflow server when the container launches
CMD mlflow server \
    --backend-store-uri ${MLFLOW_TRACKING_URI} \
    --default-artifact-root file:/mnt/mlflow \
    --host 0.0.0.0