"""Experimentos reproducibles para el Entregable II de LogisTech.

La evaluacion respeta el orden temporal y ajusta cada transformacion solamente
con los datos disponibles en entrenamiento. El modo rapido usa una muestra
temporalmente distribuida para que el notebook termine en Colab CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import sqrt
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
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
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR


@dataclass(frozen=True)
class ExperimentConfig:
    random_state: int = 42
    max_rows: int | None = 15_000
    cv_splits: int = 3
    bootstrap_iterations: int = 500
    max_categories: int = 20
    min_frequency: int = 20


def temporal_sample(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    max_rows: int | None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Reduce costo sin destruir el orden ni ignorar extremos temporales."""
    if max_rows is None or len(X) <= max_rows:
        return X.reset_index(drop=True), y.reset_index(drop=True), dates.reset_index(drop=True)
    positions = np.linspace(0, len(X) - 1, max_rows, dtype=int)
    return (
        X.iloc[positions].reset_index(drop=True),
        y.iloc[positions].reset_index(drop=True),
        dates.iloc[positions].reset_index(drop=True),
    )


def split_train_validation_test(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
) -> dict[str, Any]:
    """Particion cronologica 60/20/20 con comprobaciones explicitas."""
    n = len(X)
    train_end, validation_end = int(n * 0.60), int(n * 0.80)
    if train_end < 100 or validation_end <= train_end or validation_end >= n:
        raise ValueError("No hay suficientes observaciones para la particion 60/20/20.")
    parts = {
        "X_train": X.iloc[:train_end],
        "y_train": y.iloc[:train_end],
        "d_train": dates.iloc[:train_end],
        "X_validation": X.iloc[train_end:validation_end],
        "y_validation": y.iloc[train_end:validation_end],
        "d_validation": dates.iloc[train_end:validation_end],
        "X_test": X.iloc[validation_end:],
        "y_test": y.iloc[validation_end:],
        "d_test": dates.iloc[validation_end:],
    }
    return parts


def make_preprocessor(X: pd.DataFrame, config: ExperimentConfig) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric = X.columns.difference(categorical).tolist()
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=config.min_frequency,
                                max_categories=config.max_categories,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


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
