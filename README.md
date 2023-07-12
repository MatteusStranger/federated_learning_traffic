# Traffic Monitoring and Route Optimization System
This script is a part of a larger system for managing traffic in a city or other large area. It uses machine learning models to predict traffic levels and select alternate routes for vehicles, with the goal of potentially reducing travel times and improving traffic flow.

## Background and Objective

The increasing volume of traffic in urban areas necessitates the development of sophisticated traffic management systems. The complexity of traffic dynamics renders traditional methods inadequate, calling for innovative solutions incorporating advanced technologies like machine learning.

The primary goal of this research is to implement federated learning to optimize traffic routing in urban areas. Federated learning, a decentralized machine learning approach, allows us to train an algorithm across multiple decentralized edge devices holding local data samples, without the need to exchange them. This approach is particularly useful for traffic management, where data is naturally scattered across different geographic locations.

The project employs the Simulation of Urban Mobility (SUMO), an open-source, highly portable traffic simulation package, to generate traffic data under various scenarios. Machine learning models are trained using this data to predict traffic levels. The performance of these models is subsequently evaluated and logged using MLflow, an open-source platform to manage the ML lifecycle.

## Mathematical Description

The optimization problem can be mathematically formulated as follows:

Given a directed, weighted graph G = (V, E) representing the traffic network (where V is the set of vertices or intersections, and E is the set of edges or roads), the weight of each edge e ∈ E, denoted by w(e), corresponds to the traffic level on that road. The objective is to minimize the total travel time for all vehicles in the network.

Each vehicle v ∈ V has a start vertex s(v) and an end vertex t(v). Let P be the set of all simple paths from s(v) to t(v). The aim is to find a path p ∈ P that minimizes the sum of the weights of its constituent edges. This is essentially a shortest path problem and can be solved using Dijkstra's algorithm or the A* algorithm.

However, the weights of the edges are not static but depend on the traffic level, which can change over time. Therefore, the problem becomes a dynamic shortest path problem. This is where machine learning comes into play. We use regression models to predict the traffic level on the roads and update the weights of the edges accordingly.

The traffic level on each road is classified into three categories: Low, Medium, and High. This is a multiclass classification problem and can be solved using algorithms like Decision Trees, Random Forests, or Support Vector Machines.

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

## Libraries Used

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

## Implementation

The project is implemented in Python and requires Docker for deployment. The `Dockerfile` and `docker-compose.yml` files are provided for setting up the Docker environment.

SUMO is used to simulate traffic scenarios, generate traffic data, and visualize the traffic flow. The Python script `main.py` runs the SUMO simulations, trains the machine learning models, evaluates their performance, and logs the results to MLflow.

## Results and Evaluation

The performance of the models is evaluated using various metrics. For regression models (predicting traffic levels), the metrics include Mean Squared Error (MSE), Mean Absolute Error (MAE), Mean Squared Logarithmic Error (MSLE), Median Absolute Error (MedAE), and R^2 score. For classification models (classifying traffic levels), the metrics include accuracy, F1 score, precision, recall, and ROC AUC score.

The experiment results are logged to MLflow and can be viewed in the MLflow UI. The models and their performance metrics can be compared to choose the best model for traffic level prediction.

## Future Work

Future work includes improving the accuracy of traffic level prediction and traffic level classification, exploring other machine learning algorithms and features, and integrating real-time traffic data.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. The project requires Docker and Docker Compose for deployment. Please refer to the `Dockerfile` and `docker-compose.yml` for the configuration details.

# Authors
* Matteus Vargas Simão da Silva
* Dr. Luiz Fernando Bittencourt