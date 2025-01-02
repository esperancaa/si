import numpy as np
import sys
sys.path.append("/Users/utilizador/Documents/GitHub/si/src")
from si.model_selection.cross_validate import k_fold_cross_validation
from si.io.csv_file import read_csv
from si.data.dataset import Dataset
from si.models.logistic_regression import LogisticRegression
from si.model_selection.split import train_test_split
from typing import Callable, Tuple, Dict, Any


def randomized_search_cv(
    model,
    dataset,
    hyperparameter_grid: Dict[str, Tuple],
    scoring: Callable = None,
    cv: int = 5,
    n_iter: int = None,
) -> Dict[str, Any]:
    """
    Realiza busca aleatória de hiperparâmetros com validação cruzada.

    Parâmetros:
    ----------
    model : Modelo
        O modelo a ser avaliado.
    dataset : Dataset
        O dataset para validação cruzada.
    hyperparameter_grid : Dict[str, Tuple]
        Dicionário com os hiperparâmetros e seus possíveis valores.
    scoring : Callable
        Função de avaliação para o modelo.
    cv : int
        Número de folds para validação cruzada.
    n_iter : int
        Número de combinações aleatórias de hiperparâmetros a serem testadas.

    Retorna:
    -------
    results : Dict[str, Any]
        Resultados da busca aleatória, incluindo as pontuações e os melhores hiperparâmetros.
    """
    # Validação dos parâmetros
    if n_iter is None or n_iter <= 0:
        raise ValueError("O número de iterações (n_iter) deve ser maior que zero.")
    if not hyperparameter_grid or not isinstance(hyperparameter_grid, dict):
        raise ValueError("O grid de hiperparâmetros deve ser um dicionário válido.")
    if cv <= 1:
        raise ValueError("O número de folds (cv) deve ser maior que 1.")

    # Verifica se os hiperparâmetros existem no modelo
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Modelo {model} não possui o hiperparâmetro '{parameter}'.")

    results = {'scores': [], 'hyperparameters': []}

    for _ in range(n_iter):
        # Seleção aleatória de valores para os hiperparâmetros
        parameters = {key: np.random.choice(values) for key, values in hyperparameter_grid.items()}
        for key, value in parameters.items():
            setattr(model, key, value)  # Define o hiperparâmetro no modelo

        # Validação cruzada
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # Armazena os resultados
        results['scores'].append(np.mean(score))
        results['hyperparameters'].append(parameters)

    # Identifica os melhores hiperparâmetros
    best_index = np.argmax(results['scores'])
    results['best_hyperparameters'] = results['hyperparameters'][best_index]
    results['best_score'] = results['scores'][best_index]

    return results
