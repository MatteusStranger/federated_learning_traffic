import pandas as pd
import itertools

# Lista de valores para cada variável
simulation_times = [50, 100, 150, 200]
sumo_config_files = ["LuSTScenario\\scenario\\dua.actuated.sumocfg",
                     "LuSTScenario\\scenario\\dua.static.sumocfg",
                     "LuSTScenario\\scenario\\due.actuated.sumocfg",
                     "LuSTScenario\\scenario\\due.static.sumocfg"]

model_grid = {
    "LinearRegression": {"normalize": [True, False]},
    "RandomForestRegressor": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]},
    "KNeighborsRegressor": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
    "GradientBoostingRegressor": {"n_estimators": [50, 100, 150], "max_depth": [None, 5, 10]},
    "AdaBoostRegressor": {"n_estimators": [50, 100, 150], "learning_rate": [0.1, 0.5, 1.0]},
    "LogisticRegression": {"penalty": ["l1", "l2"], "C": [0.1, 1, 10]},
    "RandomForestClassifier": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]},
    "DecisionTreeClassifier": {"max_depth": [None, 10, 20]},
}

task_types = ["Previsão de Tráfego Futuro", "Classificação de Tráfego", "Detecção de Anomalias", "Otimização de Semáforos",
              "Planejamento de Rotas", "Recomendação de Modos de Transporte", "Controle de Tráfego Inteligente",
              "Previsão de Acidentes", "Avaliação de Impacto de Políticas de Trânsito", "Simulação de Tráfego Inteligente",
              "Previsão de Demanda de Transporte", "Roteirização Inteligente", "Monitoramento de Qualidade do Ar",
              "Estimativa de Fluxo de Tráfego em Tempo Real", "Identificação de Padrões de Comportamento dos Motoristas",
              "Detecção de Congestionamentos e Incidentes em Tempo Real", "Controle de Veículos Autônomos",
              "Avaliação de Impacto de Mudanças na Infraestrutura", "Previsão de Demanda de Estacionamento",
              "Avaliação de Sustentabilidade Urbana"]

model_metrics_regression = ["mse", "mae", "r2"]
model_metrics_classification = ["accuracy", "f1_score", "roc_auc"]

aggregated_models = ["mode", "mean"]

# Gerar todas as combinações possíveis
all_combinations = list(itertools.product(simulation_times, sumo_config_files, model_grid.keys(), task_types, aggregated_models))
all_combinations_expanded = []

# Expandir as combinações com os parâmetros e métricas
for combination in all_combinations:
    simulation_time, sumo_config_file, model_name, task_type, aggregated_model = combination
    model_parameters = model_grid[model_name]
    model_metrics = model_metrics_regression if task_type == "Regressão" else model_metrics_classification
    for parameters in itertools.product(*model_parameters.values()):
        for metrics in itertools.permutations(model_metrics):
            all_combinations_expanded.append((simulation_time, sumo_config_file, model_name, model_parameters, task_type, parameters, aggregated_model, metrics))

# Criar DataFrame com todas as informações
columns = ["simulation_times", "sumo_config_files", "model_name", "model_parameters", "task_type", "model_parameters", "aggregated_model", "model_metrics"]
df = pd.DataFrame(all_combinations_expanded, columns=columns)

# Mostrar as primeiras 5 linhas do DataFrame
df.to_csv('experiments.csv', sep=';')
