# Use the latest Ubuntu image from DockerHub
FROM ubuntu:latest
# Set the working directory in the Docker image
WORKDIR /app
# Copy the project files to the working directory
COPY . /app/

RUN rm -rf /app/artifacts/

# Install system dependencies
RUN apt-get update && apt-get dist-upgrade -y && apt install software-properties-common -y

RUN add-apt-repository ppa:sumo/stable

RUN apt-get install -y git cmake python3 \
    python3-pip g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev \
    libgl2ps-dev python3-dev swig default-jdk maven libeigen3-dev \
    sumo sumo-tools sumo-doc

ENV SUMO_HOME="/usr/bin/sumo"

# Install Python and pip
RUN apt-get install -y python3-pip
# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the entry point for the container
ENTRYPOINT ["/bin/bash", "-c", "mlflow server --host 0.0.0.0 --backend-store-uri ${MLFLOW_TRACKING_URI} --default-artifact-root ./artifacts"]
