# Traffic Monitoring and Route Optimization System
This script is a part of a larger system for managing traffic in a city or other large area. It uses machine learning models to predict traffic levels and select alternate routes for vehicles, with the goal of potentially reducing travel times and improving traffic flow.

## Features

* Collects data from a SUMO traffic simulation.
* Validates the collected data to ensure it contains the expected columns.
* Trains a variety of machine learning models to predict traffic levels.
* Evaluates the models' performance using metrics like Mean Squared Error (MSE) and R2 Score.
* Uses the trained models to predict traffic levels.
* Selects alternate routes for each vehicle based on the traffic level predictions.
* Evaluates the performance of the new routes selected.
* Logs various metrics, parameters, and results for later analysis using MLFlow.
* Saves the trained models.

# Libraries Used

* *os* - For interacting with the OS and managing file paths.
* *mlflow* - For logging metrics and results.
* *pandas* - For data manipulation and analysis.
* *traci* - To interact with SUMO, a traffic simulation software.
* *sklearn* - For machine learning model training and evaluation.
* *networkx* - For handling and analyzing complex networks (used in route selection).
* *matplotlib* - For generating plots and visualizations.
* *joblib* - For parallel computation.

# Docker

A Dockerfile is provided to build a Docker image of the application. The Docker image is based on the latest Ubuntu image with SUMO, Python 3, pip, and other necessary dependencies installed. It sets up an MLflow server to run when the container starts.

To build the Docker image, navigate to the directory containing the Dockerfile and run the following command:

```bash
docker build -t your_image_name .
```
Then, to run the Docker container, use the command:

```bash
docker run -p 5000:5000 your_image_name
```

Make sure to replace your_image_name with whatever you want to name your Docker image.

# Docker Compose
A docker-compose.yaml file is provided to run the application and its dependencies as a collection of services. The Docker Compose file defines two services: mlflow and db. The mlflow service builds an image from the Dockerfile in the current context and sets up an environment with the necessary settings. The db service uses the latest postgres image to create a PostgreSQL database.

To use Docker Compose, navigate to the directory containing the docker-compose.yaml file and run:

```bash
docker-compose up
```

# How to Run

Please ensure that you have all the necessary libraries installed before running the script.

To run the script, you would typically use the following command in the terminal:

```bash
python main.py
```

# Authors
* Matteus Vargas Simão da Silva
* Dr. Luiz Fernando Bittencourt