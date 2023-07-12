# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install SUMO
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN wget https://github.com/eclipse/sumo/archive/v1_8_0.tar.gz \
    && tar xzf v1_8_0.tar.gz \
    && rm v1_8_0.tar.gz

WORKDIR /opt/sumo-1_8_0

RUN make -f Makefile.cvs && ./configure && make

# Set environment variables for SUMO
ENV SUMO_HOME /opt/sumo-1_8_0
ENV PATH $PATH:$SUMO_HOME/bin

# Install MLflow
RUN pip install mlflow

# Expose port for MLflow server
EXPOSE 5000

# Set the working directory back to /app
WORKDIR /app

# Run the command to start MLflow server when the container launches
CMD mlflow server \
    --backend-store-uri postgresql://username:password@localhost/mlflow \
    --default-artifact-root file:/mnt/mlflow \
    --host 0.0.0.0
