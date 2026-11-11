"""Orquestador reproducible del experimento de un objetivo.

`run_task` encadena las fases exigidas por el Entregable II: muestreo
temporal, particion 60/20/20, busqueda de hiperparametros con validacion
cruzada temporal, ajuste final con intervalo bootstrap sobre prueba,
analisis de variables por permutacion y comparacion con reduccion
dimensional (PCA y UMAP).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation import (
    feature_analysis,
    fit_best_and_test,
    reduction_comparison,
    tune_models,
)
from src.models import ExperimentConfig
from src.splits import split_train_validation_test, temporal_sample


def run_task(
    task: str,
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    config: ExperimentConfig,
) -> dict[str, Any]:
    X, y, dates = temporal_sample(X, y, dates, config.max_rows)
    parts = split_train_validation_test(X, y, dates)
    X_development = pd.concat([parts["X_train"], parts["X_validation"]])
    y_development = pd.concat([parts["y_train"], parts["y_validation"]])
    cv_results, best = tune_models(task, X_development, y_development, config)
    ranking, fitted = fit_best_and_test(task, parts, best, config)
    champion = ranking.iloc[0]["model"]
    importance = feature_analysis(
        task,
        fitted[champion],
        parts["X_test"],
        parts["y_test"],
        config.random_state,
    )
    reductions = reduction_comparison(task, parts, ranking, best, config)
    return {
        "parts": parts,
        "cv_results": cv_results,
        "ranking": ranking,
        "models": fitted,
        "feature_importance": importance,
        "reductions": reductions,
    }
