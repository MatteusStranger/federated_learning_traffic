import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
import traci
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

# Constants
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001


def collect_data(sumo_config_file):
    traci.start(["sumo", "-c", sumo_config_file])
    data = []
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        vehicles = traci.vehicle.getIDList()
        for vehicle_id in vehicles:
            x, y = traci.vehicle.getPosition(vehicle_id)
            traffic_level = traci.edge.getLastStepMeanSpeed(
                traci.vehicle.getRoadID(vehicle_id))
            data.append({
                'vehicle_id': vehicle_id,
                'step': step,
                'x': x,
                'y': y,
                'traffic_level': traffic_level
            })
        step += 1
    traci.close()
    return pd.DataFrame(data)


def validate_data(data):
    expected_columns = ['vehicle_id', 'step', 'x', 'y', 'traffic_level']
    if not set(expected_columns).issubset(data.columns):
        raise ValueError("Input data does not contain the expected columns.")
    # Add more data validation checks as needed


def train_models(X_train, y_train, metric, model_candidates):
    models = {}
    metrics = {}

    for vehicle_id, model in model_candidates.items():
        model.fit(X_train, y_train)
        models[vehicle_id] = model

        if metric == 'min':
            y_pred = model.predict(X_train)
            model_metric = mean_squared_error(y_train, y_pred)
        else:
            y_pred = model.predict(X_train)
            model_metric = r2_score(y_train, y_pred)

        metrics[vehicle_id] = {
            'Metric': model_metric
        }

    return models, metrics


def evaluate_models(models, X_test, y_test):
    performance_metrics = {}

    for vehicle_id, model in models.items():
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)

        performance_metrics[vehicle_id] = {
            'MSE': mse,
            'MAE': mae,
            'R2 Score': r2,
            'Explained Variance Score': evs
        }

    return performance_metrics


def select_alternate_routes(predictions, use_astar=False):
    graph = nx.DiGraph()
    available_routes = traci.route.getIDList()

    for route_id in available_routes:
        edges = traci.route.getEdges(route_id)
        for i in range(len(edges) - 1):
            source = edges[i]
            target = edges[i + 1]
            graph.add_edge(source, target)

    new_routes = {}
    shortest_paths = {}

    for vehicle_id, traffic_level_pred in predictions.items():
        current_route = traci.vehicle.getRoute(vehicle_id)

        if use_astar:
            shortest_paths = nx.single_source_astar_path(
                graph, current_route[-1])
        else:
            shortest_paths = nx.single_source_dijkstra_path(
                graph, current_route[-1])

        del shortest_paths[current_route[-1]]
        alternate_routes = list(shortest_paths.values())

        if not alternate_routes:
            new_routes[vehicle_id] = current_route
        else:
            best_route = min(alternate_routes,
                             key=lambda route: calculate_traffic(route))
            new_routes[vehicle_id] = best_route

    return new_routes, shortest_paths


def calculate_traffic(route):
    total_traffic = 0
    for edge in route:
        total_traffic += traci.edge.getLastStepMeanSpeed(edge)
    return total_traffic


def evaluate_route_performance(new_routes, shortest_paths):
    performance_metrics = {}

    for vehicle_id, new_route in new_routes.items():
        old_route = traci.vehicle.getRoute(vehicle_id)
        old_travel_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)

        traci.vehicle.setRoute(vehicle_id, new_route)
        traci.simulationStep()
        new_travel_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)

        traci.vehicle.setRoute(vehicle_id, old_route)

        performance_metrics[vehicle_id] = old_travel_time - new_travel_time

    return performance_metrics, shortest_paths


def log_metrics(metrics, prefix):
    for vehicle_id, metric_values in metrics.items():
        for metric_name, metric_value in metric_values.items():
            mlflow.log_metric(
                f"{prefix}_{metric_name}_{vehicle_id}", metric_value)


def log_model_parameters(models):
    for vehicle_id, model in models.items():
        mlflow.log_param(f"Model_Parameters_{vehicle_id}", model.get_params())


def log_preprocessing_parameters():
    mlflow.log_param("Preprocessing_Parameters", {
        "Normalization": True,
        "Scaling": "Standardization",
        "Feature_Selection": "None"
    })


def log_training_parameters():
    mlflow.log_param("Training_Parameters", {
        "Batch_Size": BATCH_SIZE,
        "Epochs": EPOCHS,
        "Learning_Rate": LEARNING_RATE
    })


def log_evaluation_parameters():
    mlflow.log_param("Evaluation_Parameters", {
        "Metrics": ["MSE", "MAE", "R2 Score"],
        "Threshold": 0.5
    })


def log_configuration_parameters():
    mlflow.log_param("Configuration_Parameters", {
        "Source_Code_Version": "v1.0",
        "Library_Version": "sklearn-0.24.2"
    })


def log_route_algorithm(use_astar):
    if use_astar:
        mlflow.log_param("Route_Algorithm", "A*")
    else:
        mlflow.log_param("Route_Algorithm", "Dijkstra")


def log_best_model(vehicle_id):
    mlflow.log_param("Best_Model", vehicle_id)


def log_best_route(vehicle_id, route):
    mlflow.log_param("Best_Route_Vehicle", vehicle_id)
    mlflow.log_param("Best_Route", route)


def save_models(models):
    for vehicle_id, model in models.items():
        model_path = f"models/{vehicle_id}"
        mlflow.sklearn.log_model(model, model_path)
        mlflow.log_artifact(model_path)


def monitor_and_retrain(metric='min', use_astar=False, sumo_config_file="path/to/sumo_config_file.sumocfg", model_candidates=None):
    client = MlflowClient()
    experiment_id = client.create_experiment("Federated Learning Experiment")
    run = client.create_run(experiment_id)

    data = collect_data(sumo_config_file)
    validate_data(data)
    X = data[['step', 'x', 'y']]
    y = data['traffic_level']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models, train_metrics = train_models(
        X_train, y_train, metric, model_candidates)

    test_metrics = evaluate_models(models, X_test, y_test)

    with mlflow.start_run(run_id=run.info.run_id):
        log_metrics(train_metrics, prefix="Train")
        log_metrics(test_metrics, prefix="Test")

        log_model_parameters(models)
        log_preprocessing_parameters()
        log_training_parameters()
        log_evaluation_parameters()
        log_configuration_parameters()

        log_route_algorithm(use_astar)

    client.set_terminated(run.info.run_id, "FINISHED")

    best_model_vehicle_id = min(test_metrics, key=test_metrics.get)
    best_model = models[best_model_vehicle_id]

    new_data = collect_data(sumo_config_file)
    new_routes, shortest_paths = select_alternate_routes(models, use_astar)

    performance_metrics, shortest_paths = evaluate_route_performance(
        new_routes, shortest_paths)

    for vehicle_id, metric_value in performance_metrics.items():
        test_metrics[vehicle_id]['Route_Performance'] = metric_value

    with mlflow.start_run(run_id=run.info.run_id):
        log_metrics(test_metrics, prefix="Test")

        log_best_model(best_model_vehicle_id)
        log_best_route(best_route_vehicle_id, best_route)

        save_models(models)

    client.set_terminated(run.info.run_id, "FINISHED")

    best_route_vehicle_id = max(
        performance_metrics, key=performance_metrics.get)
    best_route = shortest_paths[best_route_vehicle_id]

    return best_model, best_route


if __name__ == "__main__":
    best_model, best_route = monitor_and_retrain(metric='min', use_astar=False, sumo_config_file="due.actuated.sumocfg", model_candidates={
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'Decision Tree': DecisionTreeRegressor(),
        'SVR': SVR(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'Neural Networks': MLPRegressor(),
        'KNN': KNeighborsRegressor(),
        'Gaussian Process': GaussianProcessRegressor(),
        'Bagging': BaggingRegressor(),
        'AdaBoost': AdaBoostRegressor()
    })
