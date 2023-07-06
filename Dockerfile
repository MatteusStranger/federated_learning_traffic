# Use the latest Python image from DockerHub
FROM python:latest

# Set the working directory in the Docker image
WORKDIR /app

# Copy the requirements.txt file into the Docker image
COPY requirements.txt .

# Install the Python libraries specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN add-apt-repository ppa:sumo/stable -y
RUN apt-get update
RUN apt install sumo sumo-tools sumo-doc -y

# Copy the rest of the code into the Docker image
COPY . .

# Set the command to run when the Docker image is started
CMD ["python", "main.py"]
