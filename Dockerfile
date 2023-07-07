# Use the latest Ubuntu image from DockerHub
FROM ubuntu:latest
# Set the working directory in the Docker image
WORKDIR /app
# Copy the project files to the working directory
COPY . /app/
# Install system dependencies
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:sumo/stable -y
# Add SUMO repository and install SUMO and related packages
RUN apt-get update && \
 apt-get install -y sumo sumo-tools sumo-doc
# Install Python and pip
RUN apt-get install -y python3-pip
# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
# Set the entry point for the container
ENV MLFLOW_SERVER_WORKERS=9
ENTRYPOINT ["/bin/bash", "-c", "mlflow server --host 0.0.0.0 --backend-store-uri ${MLFLOW_TRACKING_URI} --default-artifact-root ./artifacts"]
