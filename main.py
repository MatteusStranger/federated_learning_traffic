import pandas as pd
import numpy as np
import os
import traci
import pickle
import socket
import pickle
import socket
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score, \
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import psycopg2
import json
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import os
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os

# Defina as constantes simulation_times e sumo_config_files
simulation_times = [50, 100, 150, 200]
sumo_config_files = ["LuSTScenario\\scenario\\dua.actuated.sumocfg", "LuSTScenario\\scenario\\dua.static.sumocfg",
                     "LuSTScenario\\scenario\\due.actuated.sumocfg", "LuSTScenario\\scenario\\due.static.sumocfg"]

# Outras constantes e configurações
experiment_name = "Federated_Learning_Results"
run_name_prefix = "FL"
output_folder = "output_results"


def send_model_to_device(model, device_address):
    """
    Envia o modelo treinado para um dispositivo.

    Parameters:
    model (dict): Dicionário contendo os melhores parâmetros do modelo e o modelo serializado.
    device_address (str): Endereço do dispositivo para o qual o modelo será enviado.

    Returns:
    bool: True se o envio foi bem-sucedido, False caso contrário.
    """
    try:
        # Serializa o modelo
        model_bytes = pickle.dumps(model)

        # Estabelece a conexão com o dispositivo
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Define um tempo limite para a conexão (5 segundos)
        client_socket.settimeout(5)

        try:
            client_socket.connect((device_address, 12345))
        except ConnectionRefusedError:
            print("Erro: Conexão recusada. Certifique-se de que o dispositivo está disponível e aguardando conexões.")
            client_socket.close()
            return False

        # Envia os dados do modelo
        total_sent = 0
        while total_sent < len(model_bytes):
            sent = client_socket.send(model_bytes[total_sent:])
            if sent == 0:
                raise RuntimeError(
                    "Erro: A conexão foi fechada antes de enviar todos os dados.")
            total_sent += sent

        # Fecha a conexão
        client_socket.close()

        return True

    except socket.timeout:
        print("Tempo limite de conexão excedido. O dispositivo pode estar indisponível.")
        return False

    except Exception as e:
        print("Erro ao enviar o modelo para o dispositivo:", e)
        return False


def aggregate_results(results, aggregation_metric='mean'):
    """
    Agrega os resultados recebidos dos dispositivos.

    Parameters:
    results (list): Lista de dicionários contendo os resultados dos dispositivos.
    aggregation_metric (str): A métrica de agregação a ser utilizada. Pode ser 'mean', 'median', 'min' ou 'max'.

    Returns:
    dict: Dicionário contendo os resultados agregados ou None se nenhum resultado estiver disponível.
    """
    if not results:
        print("Nenhum resultado foi recebido para agregação.")
        return None

    # Verifica se os resultados possuem a mesma tarefa (regressão ou classificação)
    task_types = set(result['task_type'] for result in results)
    if len(task_types) > 1:
        print("Erro: Os resultados possuem tarefas diferentes. Não é possível realizar a agregação.")
        return None

    aggregated_results = {
        'metrics': {},
        'task_type': task_types.pop()
    }

    # Verifica as métricas disponíveis em cada resultado
    available_metrics = set(result['metrics'].keys() for result in results)
    common_metrics = set.intersection(*available_metrics)

    # Agrega as métricas comuns
    for metric in common_metrics:
        values = [result['metrics'][metric] for result in results]

        if aggregation_metric == 'mean':
            aggregated_results['metrics'][metric] = np.mean(values)
        elif aggregation_metric == 'median':
            aggregated_results['metrics'][metric] = np.median(values)
        elif aggregation_metric == 'min':
            aggregated_results['metrics'][metric] = np.min(values)
        elif aggregation_metric == 'max':
            aggregated_results['metrics'][metric] = np.max(values)
        else:
            print(
                "Erro: Métrica de agregação inválida. Escolha 'mean', 'median', 'min' ou 'max'.")
            return None

    return aggregated_results


def generate_tables_and_plots(results, output_folder):
    """
    Generate relevant tables and plots from the results of federated learning experiments.

    Parameters:
    results (dict): Dictionary containing results for each task.
    output_folder (str): Folder where the tables and plots will be saved.

    Returns:
    None.
    """
    for task, task_results in results.items():
        print(f"Generating tables and plots for {task}...")
        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(task_results)

        # Generate tables
        table_filename = f"{task}_table.csv"
        table_path = os.path.join(output_folder, table_filename)
        df.to_csv(table_path, index=False)

        # Generate relevant plots
        if task == "Previsão de Tráfego Futuro":
            plot_filename = f"{task}_metrics_plot.png"
            plot_path = os.path.join(output_folder, plot_filename)
            generate_line_plot(df, x_column='Scenario', y_columns=['MSE', 'MAE', 'R-squared'],
                               x_label='Scenario', y_labels=['Mean Squared Error', 'Mean Absolute Error', 'R-squared'],
                               title=f"Metrics for {task}", save_path=plot_path,
                               line_styles=['-', '--', ':'], colors=['blue', 'orange', 'green'],
                               markers=['o', 's', '^'])

        elif task == "Classificação de Tráfego":
            plot_filename = f"{task}_metrics_plot.png"
            plot_path = os.path.join(output_folder, plot_filename)
            generate_line_plot(df, x_column='Scenario', y_columns=['Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC-AUC'],
                               x_label='Scenario', y_labels=['Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC-AUC'],
                               title=f"Metrics for {task}", save_path=plot_path,
                               line_styles=['-', '--', ':', '-.', ':'], colors=['blue', 'orange', 'green', 'red', 'purple'],
                               markers=['o', 's', '^', 'x', 'D'])

        # Adicione outros gráficos e tabelas relevantes para cada tarefa conforme necessário
        elif task == "Detecção de Anomalias":
            plot_filename = f"{task}_bar_chart.png"
            plot_path = os.path.join(output_folder, plot_filename)
            generate_bar_chart(df, x_column='Scenario', y_columns=['Anomaly_Count'],
                               x_label='Scenario', y_label='Anomaly Count',
                               title=f"Anomaly Count for {task}", save_path=plot_path,
                               colors=['purple'])

        elif task == "Otimização de Semáforos":
            plot_filename = f"{task}_scatter_plot.png"
            plot_path = os.path.join(output_folder, plot_filename)
            generate_scatter_plot(df, x_column='Optimization_Parameter', y_column='Traffic_Flow',
                                  x_label='Optimization Parameter', y_label='Traffic Flow',
                                  title=f"Traffic Flow for {task}", save_path=plot_path,
                                  marker='o', color='blue', size_column='Traffic_Count')

        # Adicione outras configurações de gráficos e tabelas para cada tarefa conforme necessário

        print(f"Tables and plots generated for {task}.")


def generate_line_plot(data, x_column, y_columns, x_label, y_labels, title, save_path,
                       line_styles=None, colors=None, markers=None):
    """
    Generate a line plot with multiple y-axes.

    Parameters:
    data (pd.DataFrame): Data for the plot.
    x_column (str): Column name for the x-axis values.
    y_columns (list): List of column names for the y-axis values.
    x_label (str): Label for the x-axis.
    y_labels (list): List of labels for the y-axes.
    title (str): Title of the plot.
    save_path (str): File path to save the plot.
    line_styles (list): List of line styles for each line in the plot.
    colors (list): List of colors for each line in the plot.
    markers (list): List of markers for each line in the plot.

    Returns:
    None.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    if not line_styles:
        line_styles = ['-'] * len(y_columns)
    if not colors:
        colors = ['blue'] * len(y_columns)
    if not markers:
        markers = ['o'] * len(y_columns)

    ax2 = ax1.twinx()
    axes = [ax1, ax2]

    for ax, column, ylabel, linestyle, color, marker in zip(axes, y_columns, y_labels, line_styles, colors, markers):
        ax.plot(data[x_column], data[column], label=ylabel,
                linestyle=linestyle, color=color, marker=marker)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='y')

    plt.xlabel(x_label)
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_bar_chart(data, x_column, y_columns, x_label, y_label, title, save_path, colors=None):
    """
    Generate a stacked bar chart.

    Parameters:
    data (pd.DataFrame): Data for the chart.
    x_column (str): Column name for the x-axis values.
    y_columns (list): List of column names for the y-axis values (stacked).
    x_label (str): Label for the x-axis.
    y_label (str): Label for the y-axis.
    title (str): Title of the chart.
    save_path (str): File path to save the chart.
    colors (list): List of colors for the bars in the chart.

    Returns:
    None.
    """
    plt.figure(figsize=(10, 6))
    if not colors:
        colors = plt.cm.tab20.colors[:len(y_columns)]

    bottom = None
    for i, column in enumerate(y_columns):
        plt.bar(data[x_column], data[column], label=column,
                color=colors[i], bottom=bottom)
        if not bottom:
            bottom = data[column]
        else:
            bottom += data[column]

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_scatter_plot(data, x_column, y_column, x_label, y_label, title, save_path, marker='o', color='blue', size_column=None):
    """
    Generate a scatter plot with optional variable-sized markers.

    Parameters:
    data (pd.DataFrame): Data for the plot.
    x_column (str): Column name for the x-axis values.
    y_column (str): Column name for the y-axis values.
    x_label (str): Label for the x-axis.
    y_label (str): Label for the y-axis.
    title (str): Title of the plot.
    save_path (str): File path to save the plot.
    marker (str): Marker style for the scatter plot.
    color (str): Color of the markers.
    size_column (str): Column name for the variable-sized markers (optional).

    Returns:
    None.
    """
    plt.figure(figsize=(10, 6))
    if size_column:
        sizes = data[size_column] * 100
    else:
        sizes = 100

    plt.scatter(data[x_column], data[y_column],
                marker=marker, c=color, s=sizes)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_model_local(features, target, model_name='LinearRegression', hyperparameters=None, cv=5):
    """
    Treina um modelo de aprendizado de máquina localmente.

    Parameters:
    features (pd.DataFrame): DataFrame contendo as features de treinamento.
    target (pd.Series): Série contendo o target de treinamento.
    model_name (str): Nome do modelo a ser utilizado. Opções: 'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet',
                      'RandomForestRegressor', 'GradientBoostingRegressor', 'SVR', 'KNeighborsRegressor',
                      'GaussianNB', 'MLPRegressor', 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor',
                      'DecisionTreeRegressor'.
    hyperparameters (dict): Dicionário com os hiperparâmetros a serem ajustados através de validação cruzada. Default é None.
    cv (int): Número de folds para validação cruzada. Default é 5.

    Returns:
    dict: Dicionário contendo os melhores parâmetros do modelo e a pontuação do modelo.
    """
    available_models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'RandomForestRegressor': RandomForestRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'SVR': SVR(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'GaussianNB': GaussianNB(),
        'MLPRegressor': MLPRegressor(),
        'XGBRegressor': XGBRegressor(),
        'LGBMRegressor': LGBMRegressor(),
        'CatBoostRegressor': CatBoostRegressor(),
        'DecisionTreeRegressor': DecisionTreeRegressor()
    }

    if model_name not in available_models:
        raise ValueError("Invalid model_name. Available options: {}".format(
            ', '.join(available_models.keys())))

    model = available_models[model_name]

    if hyperparameters is not None:
        grid_search = GridSearchCV(model, hyperparameters, cv=cv)
        grid_search.fit(features, target)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
    else:
        model.fit(features, target)
        best_params = model.get_params()
        best_score = model.score(features, target)

    return {'best_params': best_params, 'best_score': best_score}


def split_data_for_devices(data, num_devices):
    """
    Divide os dados coletados para enviar para os dispositivos.

    Parameters:
    data (pd.DataFrame): DataFrame contendo os dados coletados.
    num_devices (int): Número de dispositivos para os quais os dados serão divididos.

    Returns:
    List[pd.DataFrame]: Uma lista de DataFrames contendo os dados para cada dispositivo.
    """
    # Organiza os dados por tipo de veículo
    data_by_vehicle_type = {}
    for vehicle_type, df in data.groupby('vehicle_type'):
        data_by_vehicle_type[vehicle_type] = df

    data_for_devices = []

    # Distribui igualmente os tipos de veículos entre os dispositivos
    num_types_per_device = len(data_by_vehicle_type) // num_devices
    remaining_types = len(data_by_vehicle_type) % num_devices

    device_idx = 0
    for vehicle_type, df in data_by_vehicle_type.items():
        if num_types_per_device > 0:
            start_idx = device_idx * num_types_per_device
            end_idx = start_idx + num_types_per_device
            types_for_device = list(data_by_vehicle_type.keys())[
                start_idx:end_idx]
        else:
            types_for_device = [vehicle_type]

        data_device = pd.DataFrame(columns=data.columns)
        for type_ in types_for_device:
            data_device = data_device.append(data_by_vehicle_type[type_])

        data_for_devices.append(data_device)
        device_idx += 1

    return data_for_devices


def save_results_to_mlflow_and_postgresql(results, task_type, model_name, model_parameters, model_metrics):
    """
    Salva os resultados no MLflow e no PostgreSQL.

    Parameters:
    results (dict): Dicionário contendo os resultados agregados.
    task_type (str): O tipo de tarefa (regressão ou classificação).
    model_name (str): O nome do modelo.
    model_parameters (dict): Dicionário contendo os hiperparâmetros do modelo.
    model_metrics (dict): Dicionário contendo as métricas do modelo.

    Returns:
    bool: True se os resultados foram salvos com sucesso, False caso contrário.
    """
    try:
        # Inicia uma execução do MLflow
        with mlflow.start_run():
            # Salva os hiperparâmetros do modelo no MLflow
            for param_name, param_value in model_parameters.items():
                mlflow.log_param(param_name, param_value)

            # Salva as métricas do modelo no MLflow
            for metric_name, metric_value in model_metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Salva o tipo de tarefa no MLflow
            mlflow.log_param("task_type", task_type)

            # Salva o nome do modelo no MLflow
            mlflow.log_param("model_name", model_name)

            # Conecta ao PostgreSQL e salva os resultados
            connection = psycopg2.connect(
                user="user",
                password="password",
                host="localhost",
                port="5432",
                database="results_db"
            )
            cursor = connection.cursor()

            # Cria a tabela se não existir
            create_table_query = '''
                CREATE TABLE IF NOT EXISTS model_results (
                    model_id SERIAL PRIMARY KEY,
                    model_name VARCHAR(255),
                    task_type VARCHAR(50),
                    parameters JSONB,
                    metrics JSONB
                );
            '''
            cursor.execute(create_table_query)

            # Monta o JSON dos parâmetros e métricas
            parameters_json = json.dumps(model_parameters)
            metrics_json = json.dumps(model_metrics)

            # Verifica se o modelo já foi salvo na tabela
            select_query = '''
                SELECT model_name, parameters, metrics FROM model_results
                WHERE model_name = %s AND parameters = %s AND metrics = %s
            '''
            record_to_select = (model_name, parameters_json, metrics_json)
            cursor.execute(select_query, record_to_select)
            existing_records = cursor.fetchall()

            if existing_records:
                print("Erro: Este modelo já foi salvo na tabela.")
                cursor.close()
                connection.close()
                return False

            # Insere os resultados na tabela
            insert_query = '''
                INSERT INTO model_results (model_name, task_type, parameters, metrics)
                VALUES (%s, %s, %s, %s)
            '''
            record_to_insert = (model_name, task_type,
                                parameters_json, metrics_json)
            cursor.execute(insert_query, record_to_insert)

            connection.commit()
            cursor.close()
            connection.close()

            return True

    except Exception as e:
        print("Erro ao salvar os resultados:", e)
        return False


def generate_bar_chart(x_values, y_values, title='', x_label='', y_label='', bar_style='-', bar_color='blue', save_to_file=None):
    """
    Gera um gráfico de barras.

    Parameters:
    x_values (list): Lista contendo os valores do eixo x.
    y_values (list): Lista contendo os valores do eixo y.
    title (str): Título do gráfico (opcional).
    x_label (str): Rótulo do eixo x (opcional).
    y_label (str): Rótulo do eixo y (opcional).
    bar_style (str): Estilo das barras no gráfico (opcional). Pode ser '-', '--', '-.', ':', entre outros.
    bar_color (str): Cor das barras no gráfico (opcional).
    save_to_file (str): Caminho para salvar o gráfico em um arquivo PNG (opcional).

    Returns:
    bool: True se o gráfico foi gerado com sucesso, False caso contrário.
    """
    try:
        plt.bar(x_values, y_values, bar_style, color=bar_color)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        if save_to_file:
            plt.savefig(save_to_file)
        else:
            plt.show()

        return True

    except Exception as e:
        print("Erro ao gerar o gráfico:", e)
        return False


def generate_line_plot(x_values, y_values, title='', x_label='', y_label='', line_style='-', legend=None, save_to_file=None):
    """
    Gera um gráfico de linha.

    Parameters:
    x_values (list): Lista contendo os valores do eixo x.
    y_values (list): Lista contendo os valores do eixo y.
    title (str): Título do gráfico (opcional).
    x_label (str): Rótulo do eixo x (opcional).
    y_label (str): Rótulo do eixo y (opcional).
    line_style (str): Estilo da linha no gráfico (opcional). Pode ser '-', '--', '-.', ':', entre outros.
    legend (str): Legenda do gráfico (opcional).
    save_to_file (str): Caminho para salvar o gráfico em um arquivo PNG (opcional).

    Returns:
    bool: True se o gráfico foi gerado com sucesso, False caso contrário.
    """
    try:
        plt.plot(x_values, y_values, line_style, label=legend)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        if legend:
            plt.legend()
        if save_to_file:
            plt.savefig(save_to_file)
        else:
            plt.show()

        return True

    except Exception as e:
        print("Erro ao gerar o gráfico:", e)
        return False


def evaluate_model(model, features, target):
    """
    Avalia o desempenho do modelo usando várias métricas de avaliação.

    Parameters:
    model: Modelo treinado a ser avaliado.
    features (pd.DataFrame): DataFrame contendo as features de teste.
    target (pd.Series): Série contendo o target de teste.

    Returns:
    dict: Dicionário contendo as métricas de avaliação do modelo.
    """
    # Obtém as previsões do modelo
    predictions = model.predict(features)

    # Verifica o tipo de tarefa (regressão ou classificação)
    if isinstance(predictions[0], (int, float)):
        # Regressão
        mse = mean_squared_error(target, predictions)
        mae = mean_absolute_error(target, predictions)
        msle = mean_squared_log_error(target, predictions)
        r2 = r2_score(target, predictions)

        return {'MSE': mse, 'MAE': mae, 'MSLE': msle, 'R2': r2}

    else:
        # Classificação
        accuracy = accuracy_score(target, predictions)
        precision = precision_score(target, predictions, average='macro')
        recall = recall_score(target, predictions, average='macro')
        f1 = f1_score(target, predictions, average='macro')
        confusion = confusion_matrix(target, predictions)

        return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Confusion Matrix': confusion}


def send_gradients_to_server(gradients, server_address, server_port):
    """
    Send the gradients to the central server.

    Parameters:
    gradients (dict): Dictionary containing gradients of the model parameters.
    server_address (str): IP address of the central server.
    server_port (int): Port number of the central server.

    Returns:
    None
    """
    # Serializar os gradientes usando pickle
    serialized_gradients = pickle.dumps(gradients)

    # Iniciar a conexão com o servidor
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_address, server_port))

    # Enviar os gradientes serializados para o servidor
    client_socket.sendall(serialized_gradients)

    # Fechar a conexão
    client_socket.close()


def collect_data(sumo_config_file, simulation_time):
    """
    Coleta dados de tráfego de uma simulação SUMO.

    Parameters:
    sumo_config_file (str): Caminho para o arquivo de configuração do SUMO.
    simulation_time (int): Tempo de duração da simulação em segundos.

    Returns:
    pd.DataFrame: DataFrame contendo os dados coletados.
    """
    try:
        import traci
        from sumolib import checkBinary
    except ImportError:
        raise ImportError(
            "SUMO and traci modules are required for data collection. Make sure SUMO is installed.")

    if not os.path.isfile(sumo_config_file):
        raise ValueError(
            f"Could not find SUMO configuration file: {sumo_config_file}")

    data = []
    step = 0

    sumo_binary = checkBinary('sumo')
    traci.start([sumo_binary, "-c", sumo_config_file])

    while step < simulation_time:
        traci.simulationStep()

        # Obter informações sobre os veículos ativos na simulação
        vehicles = traci.vehicle.getIDList()

        for vehicle_id in vehicles:
            try:
                # Verificar se o veículo ainda está ativo na simulação
                if traci.vehicle.getStopState(vehicle_id) == 0:
                    x, y = traci.vehicle.getPosition(vehicle_id)
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    lane_id = traci.vehicle.getLaneID(vehicle_id)
                    route_id = traci.vehicle.getRouteID(vehicle_id)

                    # Informações sobre o ambiente e tráfego
                    num_vehicles_on_lane = traci.lane.getLastStepVehicleNumber(
                        lane_id)
                    lane_length = traci.lane.getLength(lane_id)
                    traffic_density = num_vehicles_on_lane / lane_length

                    waiting_time = traci.vehicle.getAccumulatedWaitingTime(
                        vehicle_id)

                    # Informações sobre o tipo de veículo
                    vehicle_type = traci.vehicle.getTypeID(vehicle_id)

                    data.append({
                        'vehicle_id': vehicle_id,
                        'step': step,
                        'x': x,
                        'y': y,
                        'speed': speed,
                        'lane_id': lane_id,
                        'route_id': route_id,
                        'traffic_density': traffic_density,
                        'waiting_time': waiting_time,
                        'vehicle_type': vehicle_type
                    })
            except traci.TraCIException:
                # Lidar com exceções causadas por veículos que já deixaram a simulação
                continue

        step += 1

    traci.close()
    return pd.DataFrame(data)


def send_model_to_server(model, server_address, server_port):
    """
    Send the trained model to the central server.

    Parameters:
    model: Trained machine learning model.
    server_address (str): IP address of the central server.
    server_port (int): Port number of the central server.

    Returns:
    None
    """
    # Serializar o modelo treinado usando pickle
    serialized_model = pickle.dumps(model)

    # Iniciar a conexão com o servidor
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_address, server_port))

    # Enviar o modelo serializado para o servidor
    client_socket.sendall(serialized_model)

    # Fechar a conexão
    client_socket.close()


def receive_updated_model_parameters(server_address, server_port):
    """
    Receive the updated model parameters from the central server.

    Parameters:
    server_address (str): IP address of the central server.
    server_port (int): Port number of the central server.

    Returns:
    dict: Dictionary containing the updated model parameters.
    """
    # Iniciar a conexão com o servidor
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_address, server_port))
    server_socket.listen(1)

    # Aguardar a conexão do dispositivo local
    client_socket, client_address = server_socket.accept()

    # Receber os dados enviados pelo dispositivo local
    data = client_socket.recv(4096)

    # Fechar a conexão
    server_socket.close()

    # Desserializar os dados recebidos usando pickle
    updated_model_params = pickle.loads(data)

    return updated_model_params


def aggregate_models(updated_models, aggregated_model):
    """
    Aggregates the updated models received from devices.

    Parameters:
    updated_models (list): List of updated models received from devices.
    aggregated_model (str): Type of aggregation (mean or mode).

    Returns:
    dict: The aggregated model.
    """
    if aggregated_model == "mean":
        # Aggregation by calculating the mean of model weights
        if len(updated_models) == 0:
            raise ValueError("No updated models received for aggregation.")

        aggregated_model = {}
        for model in updated_models:
            for param_name, param_value in model.items():
                if param_name not in aggregated_model:
                    aggregated_model[param_name] = []
                aggregated_model[param_name].append(param_value)

        for param_name, param_values in aggregated_model.items():
            aggregated_model[param_name] = np.mean(param_values, axis=0)

    elif aggregated_model == "mode":
        # Aggregation by selecting the most common model
        if len(updated_models) == 0:
            raise ValueError("No updated models received for aggregation.")

        aggregated_model = {}
        for model in updated_models:
            model_str = str(model)
            if model_str not in aggregated_model:
                aggregated_model[model_str] = 0
            aggregated_model[model_str] += 1

        most_common_model = max(aggregated_model, key=aggregated_model.get)
        aggregated_model = eval(most_common_model)

    else:
        raise ValueError(
            "Invalid aggregated_model type. Use 'mean' or 'mode'.")

    return aggregated_model


def federated_learning_coordinator():
    """
    Coordinating function for Federated Learning.

    Returns:
    None.
    """
    # Dicionário para armazenar os resultados das tarefas
    results = {}

    # Executar tarefas em paralelo usando ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        for simulation_time, sumo_config_file in product(simulation_times, sumo_config_files):
            # Coletar dados de tráfego
            data = collect_data(sumo_config_file, simulation_time)

            # Dividir dados entre os dispositivos
            device_data = split_data_for_devices(data)

            # Treinar modelos em cada dispositivo
            device_results = {}
            for device_id, device_data in device_data.items():
                device_results[device_id] = executor.submit(
                    train_model_local, device_data)

            # Agregar modelos no servidor central
            aggregated_model = aggregate_models(device_results)

            # Enviar o modelo agregado para os dispositivos
            for device_id in device_results:
                send_model_to_device(aggregated_model, device_id)

            # Avaliar o desempenho do modelo em cada dispositivo
            for device_id in device_results:
                device_results[device_id] = evaluate_model(
                    device_results[device_id], device_data[device_id])

            # Agregar os resultados das tarefas
            results = aggregate_results(results, device_results)

    # Salvar os resultados no MLflow e PostgreSQL
    save_results_to_mlflow_and_postgresql(
        results, experiment_name, run_name_prefix, output_folder)


df = pd.read_csv('experiments.csv')


# Loop para percorrer cada linha da tabela df
for index, row in df.iterrows():
    # Extrair os valores das colunas da tabela para cada experimento
    simulation_time = row['simulation_times']
    sumo_config_file = row['sumo_config_files']
    model_grid = row['model_grid']
    task_type = row['task_type']
    model_name = row['model_name']
    model_parameters = row['model_parameters']
    model_names = row['model_names']
    updated_models = row['updated_models']
    aggregated_model = row['aggregated_model']
    devices = row['devices']
    model_metrics = row['model_metrics']

    # Realizar o experimento com os valores da linha atual
    federated_learning_coordinator(simulation_time, sumo_config_file, model_grid, task_type, model_name,
                                   model_parameters, model_names, updated_models, aggregated_model, devices, model_metrics)
