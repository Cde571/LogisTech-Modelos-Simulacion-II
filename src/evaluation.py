"""Metricas, validacion cruzada temporal, bootstrap y analisis complementario.

Las metricas estan separadas para no acoplar la busqueda con la evaluacion
final. La busqueda `tune_models` usa TimeSeriesSplit y reporta la media y
el intervalo de confianza aproximado entre pliegues. La evaluacion final en
`fit_best_and_test` reajusta sobre entrenamiento+validacion y mide sobre
prueba, agregando el intervalo bootstrap de la metrica principal.

`feature_analysis` cuantifica la contribucion de cada variable observable
mediante importancia por permutacion en el holdout de validacion.
`reduction_comparison` compara el pipeline con PCA que retiene 95% de
varianza y con UMAP a diez dimensiones, siguiendo lo exigido en el reporte.
"""

from __future__ import annotations

from math import sqrt
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.encoders import make_preprocessor
from src.models import ExperimentConfig, model_catalog


def _regression_metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, prediction),
        "RMSE": root_mean_squared_error(y_true, prediction),
        "R2": r2_score(y_true, prediction),
    }


def _classification_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def _classification_metrics(
    y_true: pd.Series, prediction: np.ndarray, score: np.ndarray
) -> dict[str, float]:
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "F1": f1_score(y_true, prediction, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, score),
    }


def _primary_value(task: str, metrics: dict[str, float]) -> float:
    return -metrics["MAE"] if task == "regression" else metrics["ROC_AUC"]


def _fit_and_metrics(
    task: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> dict[str, float]:
    prediction = pipeline.predict(X)
    if task == "regression":
        return _regression_metrics(y, prediction)
    return _classification_metrics(y, prediction, _classification_scores(pipeline, X))


def tune_models(
    task: str,
    X_development: pd.DataFrame,
    y_development: pd.Series,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Busqueda pequena pero completa con CV temporal e IC entre pliegues."""
    rows: list[dict[str, Any]] = []
    best: dict[str, dict[str, Any]] = {}
    splitter = TimeSeriesSplit(n_splits=config.cv_splits)
    catalog = model_catalog(task, config.random_state)

    for family, (estimator, grid) in catalog.items():
        family_best_score = -np.inf
        for params in ParameterGrid(grid):
            fold_scores: list[float] = []
            fold_metric_rows: list[dict[str, float]] = []
            started = perf_counter()
            for train_index, validation_index in splitter.split(X_development):
                X_train = X_development.iloc[train_index]
                y_train = y_development.iloc[train_index]
                X_validation = X_development.iloc[validation_index]
                y_validation = y_development.iloc[validation_index]
                pipeline = Pipeline(
                    [
                        ("preprocess", make_preprocessor(X_train, config)),
                        ("model", clone(estimator).set_params(**params)),
                    ]
                )
                pipeline.fit(X_train, y_train)
                metrics = _fit_and_metrics(task, pipeline, X_validation, y_validation)
                fold_metric_rows.append(metrics)
                fold_scores.append(_primary_value(task, metrics))
            mean_score = float(np.mean(fold_scores))
            standard_error = float(np.std(fold_scores, ddof=1) / sqrt(len(fold_scores)))
            row: dict[str, Any] = {
                "model": family,
                "params": params,
                "primary_cv_mean": mean_score,
                "primary_cv_ci95_low": mean_score - 1.96 * standard_error,
                "primary_cv_ci95_high": mean_score + 1.96 * standard_error,
                "fit_seconds": perf_counter() - started,
            }
            for metric_name in fold_metric_rows[0]:
                row[f"validation_{metric_name}"] = float(
                    np.mean([value[metric_name] for value in fold_metric_rows])
                )
            rows.append(row)
            if mean_score > family_best_score:
                family_best_score = mean_score
                best[family] = {"estimator": clone(estimator), "params": params}
    results = pd.DataFrame(rows).sort_values("primary_cv_mean", ascending=False)
    return results.reset_index(drop=True), best


def _bootstrap_interval(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    prediction: np.ndarray,
    iterations: int,
    random_state: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    for _ in range(iterations):
        index = rng.integers(0, len(y_true), len(y_true))
        sample_y = y_true[index]
        if len(np.unique(sample_y)) < 2 and set(np.unique(y_true)) == {0, 1}:
            continue
        values.append(metric(sample_y, prediction[index]))
    return tuple(np.quantile(values, [0.025, 0.975]))


def fit_best_and_test(
    task: str,
    parts: dict[str, Any],
    best: dict[str, dict[str, Any]],
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    X_development = pd.concat([parts["X_train"], parts["X_validation"]])
    y_development = pd.concat([parts["y_train"], parts["y_validation"]])
    fitted: dict[str, Pipeline] = {}
    rows: list[dict[str, Any]] = []
    for family, specification in best.items():
        selection_pipeline = Pipeline(
            [
                ("preprocess", make_preprocessor(parts["X_train"], config)),
                (
                    "model",
                    clone(specification["estimator"]).set_params(
                        **specification["params"]
                    ),
                ),
            ]
        )
        final_pipeline = Pipeline(
            [
                ("preprocess", make_preprocessor(X_development, config)),
                (
                    "model",
                    clone(specification["estimator"]).set_params(
                        **specification["params"]
                    ),
                ),
            ]
        )
        started = perf_counter()
        selection_pipeline.fit(parts["X_train"], parts["y_train"])
        train_metrics = _fit_and_metrics(
            task, selection_pipeline, parts["X_train"], parts["y_train"]
        )
        validation_metrics = _fit_and_metrics(
            task,
            selection_pipeline,
            parts["X_validation"],
            parts["y_validation"],
        )
        final_pipeline.fit(X_development, y_development)
        test_metrics = _fit_and_metrics(
            task, final_pipeline, parts["X_test"], parts["y_test"]
        )
        row: dict[str, Any] = {
            "model": family,
            "best_params": specification["params"],
            "fit_seconds": perf_counter() - started,
        }
        for split_name, metrics in (
            ("train", train_metrics),
            ("validation", validation_metrics),
            ("test", test_metrics),
        ):
            row.update({f"{split_name}_{name}": value for name, value in metrics.items()})
        if task == "regression":
            prediction = final_pipeline.predict(parts["X_test"])
            low, high = _bootstrap_interval(
                mean_absolute_error,
                parts["y_test"].to_numpy(),
                prediction,
                config.bootstrap_iterations,
                config.random_state,
            )
            row.update({"test_primary": test_metrics["MAE"], "test_ci95_low": low, "test_ci95_high": high})
        else:
            score = _classification_scores(final_pipeline, parts["X_test"])
            low, high = _bootstrap_interval(
                roc_auc_score,
                parts["y_test"].to_numpy(),
                score,
                config.bootstrap_iterations,
                config.random_state,
            )
            row.update({"test_primary": test_metrics["ROC_AUC"], "test_ci95_low": low, "test_ci95_high": high})
        fitted[family] = final_pipeline
        rows.append(row)
    order = "validation_MAE" if task == "regression" else "validation_ROC_AUC"
    ascending = task == "regression"
    return pd.DataFrame(rows).sort_values(order, ascending=ascending).reset_index(drop=True), fitted


def feature_analysis(
    task: str,
    fitted_model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    """Importancia por permutacion en variables originales, sin fuga."""
    scoring = "neg_mean_absolute_error" if task == "regression" else "roc_auc"
    sample_size = min(3_000, len(X_validation))
    sampled = np.linspace(0, len(X_validation) - 1, sample_size, dtype=int)
    result = permutation_importance(
        fitted_model,
        X_validation.iloc[sampled],
        y_validation.iloc[sampled],
        scoring=scoring,
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "feature": X_validation.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def reduction_comparison(
    task: str,
    parts: dict[str, Any],
    ranking: pd.DataFrame,
    best: dict[str, dict[str, Any]],
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Evalua PCA y UMAP con los dos mejores modelos del holdout de validacion."""
    try:
        from umap import UMAP
    except ImportError as error:
        raise ImportError("Instale umap-learn para ejecutar la reduccion no lineal.") from error

    validation_column = "validation_MAE" if task == "regression" else "validation_ROC_AUC"
    top_models = ranking.sort_values(
        validation_column, ascending=(task == "regression")
    )["model"].head(2).tolist()
    X_development = pd.concat([parts["X_train"], parts["X_validation"]])
    y_development = pd.concat([parts["y_train"], parts["y_validation"]])
    base_preprocessor = make_preprocessor(X_development, config)
    X_dev = base_preprocessor.fit_transform(X_development)
    X_test = base_preprocessor.transform(parts["X_test"])
    feature_count = X_dev.shape[1]

    pca_full = PCA(random_state=config.random_state).fit(X_dev)
    n_pca = int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), 0.95) + 1)
    n_pca = max(2, min(n_pca, feature_count))
    reducers: dict[str, Any] = {
        "PCA_95pct": PCA(n_components=n_pca, random_state=config.random_state),
        "UMAP_10d": UMAP(
            n_components=min(10, feature_count - 1),
            n_neighbors=20,
            min_dist=0.1,
            metric="euclidean",
            random_state=config.random_state,
            n_jobs=1,
        ),
    }
    rows: list[dict[str, Any]] = []
    for reducer_name, reducer in reducers.items():
        fit_limit = min(10_000, len(X_dev)) if reducer_name.startswith("UMAP") else len(X_dev)
        fit_index = np.linspace(0, len(X_dev) - 1, fit_limit, dtype=int)
        reducer.fit(X_dev[fit_index], y_development.iloc[fit_index])
        reduced_dev = reducer.transform(X_dev)
        reduced_test = reducer.transform(X_test)
        for family in top_models:
            estimator = clone(best[family]["estimator"]).set_params(**best[family]["params"])
            estimator.fit(reduced_dev, y_development)
            prediction = estimator.predict(reduced_test)
            metrics = (
                _regression_metrics(parts["y_test"], prediction)
                if task == "regression"
                else _classification_metrics(
                    parts["y_test"],
                    prediction,
                    estimator.predict_proba(reduced_test)[:, 1]
                    if hasattr(estimator, "predict_proba")
                    else estimator.decision_function(reduced_test)
                    if hasattr(estimator, "decision_function")
                    else prediction.astype(float),
                )
            )
            rows.append(
                {
                    "reduction": reducer_name,
                    "model": family,
                    "original_features": feature_count,
                    "reduced_features": reduced_dev.shape[1],
                    "reduction_pct": 100 * (1 - reduced_dev.shape[1] / feature_count),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)
