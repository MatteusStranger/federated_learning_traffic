from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, median_absolute_error, r2_score
import os
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from itertools import product
import mlflow
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def generate_report():
    """
    Generates a report from the logged MLFlow data and saves it as a CSV file.

    Returns:
    str: The filename of the report.
    """
    data = {
        "parameters": {},
        "metrics": {},
        "alternate_routes": {}
    }
    df = pd.DataFrame(data)
    report_filename = "report.csv"
    df.to_csv(report_filename)
    return report_filename


def collect_data(sumo_config_file, simulation_time):
    """
    Collects data from a SUMO traffic simulation.

    Parameters:
    sumo_config_file (str): The path to the SUMO configuration file.
    simulation_time (int): The time for which the simulation should run.

    Returns:
    pd.DataFrame: A DataFrame containing the collected data.
    """
    if not os.path.isfile(sumo_config_file):
        raise ValueError(
            f"Could not find SUMO configuration file: {sumo_config_file}")
    data = []
    step = 0
    while step < simulation_time:
        vehicles = []
        for vehicle_id in vehicles:
            data.append({
                'vehicle_id': vehicle_id,
                'step': step,
                'x': 0,
                'y': 0,
                'traffic_level': 0
            })
        step += 1
    return pd.DataFrame(data)


def evaluate_route_performance(new_routes, shortest_paths):
    """
    Evaluates the performance of the new routes compared to the shortest paths.

    Parameters:
    new_routes (dict): A dictionary mapping vehicle IDs to new routes.
    shortest_paths (dict): A dictionary mapping vehicle IDs to the shortest paths.

    Returns:
    dict: A dictionary mapping vehicle IDs to the difference in travel time between the new routes and the shortest paths.
    """
    performance_metrics = {}
    for vehicle_id, new_route in new_routes.items():
        old_route = []
        old_travel_time = 0
        new_travel_time = 0
        performance_metrics[vehicle_id] = old_travel_time - new_travel_time
    return performance_metrics, shortest_paths


simulation_times = [50, 100, 150, 200]
sumo_config_files = ["LuSTScenario\\scenario\\dua.actuated.sumocfg", "LuSTScenario\\scenario\\dua.static.sumocfg",
                     "LuSTScenario\\scenario\\due.actuated.sumocfg", "LuSTScenario\\scenario\\due.static.sumocfg"]
model_grid = {
    "LinearRegression": {"normalize": [True, False]},
    "RandomForestRegressor": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]},
}


def run_simulation_and_modeling(simulation_time, sumo_config_file):
    with mlflow.start_run():
        mlflow.log_param("simulation_time", simulation_time)
        mlflow.log_param("sumo_config_file", sumo_config_file)
        data = collect_data(sumo_config_file, simulation_time)
        mlflow.log_param("preprocessing", "None")
        train_data, test_data = train_test_split(data, test_size=0.2)
        features = train_data.drop('traffic_level', axis=1)
        target = train_data['traffic_level']
        for model_name, parameters in model_grid.items():
            for parameter in ParameterGrid(parameters):
                model = eval(model_name)(**parameter)
                model.fit(features, target)
                for param_name, param_value in parameter.items():
                    mlflow.log_param(f"{model_name}_{param_name}", param_value)
                predictions = model.predict(
                    test_data.drop('traffic_level', axis=1))
                mse = mean_squared_error(
                    test_data['traffic_level'], predictions)
                mae = mean_absolute_error(
                    test_data['traffic_level'], predictions)
                msle = mean_squared_log_error(
                    test_data['traffic_level'], predictions)
                medae = median_absolute_error(
                    test_data['traffic_level'], predictions)
                r2 = r2_score(test_data['traffic_level'], predictions)
                acc = accuracy_score(test_data['traffic_level'], predictions)
                f1 = f1_score(test_data['traffic_level'], predictions)
                precision = precision_score(
                    test_data['traffic_level'], predictions)
                recall = recall_score(test_data['traffic_level'], predictions)
                roc_auc = roc_auc_score(
                    test_data['traffic_level'], predictions)
                mlflow.log_metric(f"{model_name}_mse", mse)
                mlflow.log_metric(f"{model_name}_mae", mae)
                mlflow.log_metric(f"{model_name}_msle", msle)
                mlflow.log_metric(f"{model_name}_medae", medae)
                mlflow.log_metric(f"{model_name}_r2", r2)
                mlflow.log_metric(f"{model_name}_acc", acc)
                mlflow.log_metric(f"{model_name}_f1", f1)
                mlflow.log_metric(f"{model_name}_precision", precision)
                mlflow.log_metric(f"{model_name}_recall", recall)
                mlflow.log_metric(f"{model_name}_roc_auc", roc_auc)
                alternate_routes = [
                    "route" + str(i) for i in range(1, len(predictions) + 1)]
                for i, route in enumerate(alternate_routes):
                    mlflow.log_param(f"alternate_route_{i}", route)
                route_performance = evaluate_route_performance(
                    alternate_routes, alternate_routes)
                mlflow.log_metric("route_performance", route_performance)
                traffic_level_classification = {
                    vehicle_id: "High" if performance > 10 else "Medium" if performance > 5 else "Low"
                    for vehicle_id, performance in route_performance.items()
                }
                for vehicle_id, traffic_level in traffic_level_classification.items():
                    mlflow.log_param(
                        f"traffic_level_{vehicle_id}", traffic_level)
                    experiment_id = mlflow.create_experiment(
                        "Traffic Optimization")

                mlflow.set_experiment(experiment_id)
                mlflow.log_param("experiment_id", experiment_id)
                mlflow.log_param("experiment_name", "Traffic Optimization")
                mlflow.log_param("run_id", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        for simulation_time, sumo_config_file in product(simulation_times, sumo_config_files):
            executor.submit(run_simulation_and_modeling,
                            simulation_time, sumo_config_file)
