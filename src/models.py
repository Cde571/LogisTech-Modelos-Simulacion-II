"""Configuracion de experimentos y catalogo de modelos.

ExperimentConfig concentra los hiperparametros globales (semillas, tamanio
muestral, numero de pliegues temporales e iteraciones bootstrap) junto con los
umbrales de cardinalidad que gobiernan la codificacion diferenciada declarada
en el reporte.

model_catalog enumera las cinco familias exigidas: parametrica, no parametrica,
ensamble de arboles, red neuronal y maquina de soporte vectorial. Cada entrada
devuelve el estimador base y una malla pequenia de hiperparametros para la
busqueda con validacion cruzada temporal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR


@dataclass(frozen=True)
class ExperimentConfig:
    random_state: int = 42
    max_rows: int | None = 15_000
    cv_splits: int = 3
    bootstrap_iterations: int = 500
    min_frequency: int = 20
    low_cardinality_max: int = 30
    high_cardinality_min: int = 100


def model_catalog(task: str, random_state: int) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    if task == "regression":
        return {
            "Ridge (parametrico)": (Ridge(solver="lsqr"), {"alpha": [0.1, 10.0]}),
            "KNN (no parametrico)": (
                KNeighborsRegressor(n_jobs=-1),
                {"n_neighbors": [7, 25], "weights": ["distance"]},
            ),
            "Random Forest (ensamble)": (
                RandomForestRegressor(
                    n_estimators=120, n_jobs=-1, random_state=random_state
                ),
                {"max_depth": [12, None], "min_samples_leaf": [2]},
            ),
            "MLP (red neuronal)": (
                MLPRegressor(
                    max_iter=180,
                    early_stopping=True,
                    random_state=random_state,
                ),
                {"hidden_layer_sizes": [(48,), (64, 32)], "alpha": [0.0001]},
            ),
            "LinearSVR (SVM)": (
                LinearSVR(max_iter=6_000, random_state=random_state),
                {"C": [0.1, 1.0], "epsilon": [0.1]},
            ),
        }
    if task == "classification":
        return {
            "Logistica (parametrico)": (
                LogisticRegression(
                    max_iter=1_000, class_weight="balanced", random_state=random_state
                ),
                {"C": [0.2, 2.0]},
            ),
            "KNN (no parametrico)": (
                KNeighborsClassifier(n_jobs=-1),
                {"n_neighbors": [9, 31], "weights": ["distance"]},
            ),
            "Random Forest (ensamble)": (
                RandomForestClassifier(
                    n_estimators=120,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                ),
                {"max_depth": [12, None], "min_samples_leaf": [2]},
            ),
            "MLP (red neuronal)": (
                MLPClassifier(
                    max_iter=180,
                    early_stopping=True,
                    random_state=random_state,
                ),
                {"hidden_layer_sizes": [(48,), (64, 32)], "alpha": [0.0001]},
            ),
            "LinearSVC (SVM)": (
                LinearSVC(
                    class_weight="balanced", max_iter=6_000, random_state=random_state
                ),
                {"C": [0.1, 1.0]},
            ),
        }
    raise ValueError("task debe ser 'regression' o 'classification'.")
