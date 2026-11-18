"""Análisis individual de variables contra el objetivo.

Para cumplir con II.2.1 del Entregable II calculamos dos vistas por variable:

- **Correlación** contra el objetivo. Para predictores numéricos usamos
  Spearman, que captura relación monótona sin asumir linealidad. Para
  categóricos usamos la V de Cramér contra el objetivo binario (para
  `late_delivery`) o contra una discretización del objetivo continuo
  (cuantiles para `delivery_time_days`), siguiendo la práctica estándar
  cuando no se puede calcular una correlación numérica directa.
- **Información mutua** contra el objetivo, usando `mutual_info_classif`
  o `mutual_info_regression` de scikit-learn. Es una medida no lineal
  que no requiere suponer forma funcional y permite identificar
  variables con señal débil o nula.

El objetivo de la función es doble: dejar explícita la aportación
individual de cada predictor y producir una lista de candidatas a
eliminar por bajo aporte.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 2:
        return np.nan
    table = pd.crosstab(x[valid], y[valid])
    if table.size == 0 or min(table.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    denominator = min(table.shape) - 1
    if denominator == 0 or n == 0:
        return np.nan
    return float(np.sqrt(chi2 / (n * denominator)))


def _spearman(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 2:
        return np.nan
    coef, _ = spearmanr(x[valid], y[valid])
    return float(coef) if np.isfinite(coef) else np.nan


def _mutual_information(
    X: pd.DataFrame, y: pd.Series, task: str, random_state: int
) -> pd.Series:
    X_encoded = X.copy()
    discrete_mask: list[bool] = []
    for column in X_encoded.columns:
        series = X_encoded[column]
        if pd.api.types.is_numeric_dtype(series):
            X_encoded[column] = series.fillna(series.median())
            discrete_mask.append(False)
        else:
            codes = series.astype("category").cat.codes
            X_encoded[column] = codes.replace(-1, codes.max() + 1)
            discrete_mask.append(True)
    estimator = mutual_info_classif if task == "classification" else mutual_info_regression
    values = estimator(
        X_encoded,
        y,
        discrete_features=discrete_mask,
        random_state=random_state,
    )
    return pd.Series(values, index=X.columns, name="mutual_information")


def variable_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    random_state: int = 42,
    quantile_bins: int = 4,
) -> pd.DataFrame:
    """Combina correlación e información mutua por variable.

    Devuelve un DataFrame con columnas `feature`, `type` (numeric/categorical),
    `correlation` (Spearman o V de Cramér), `mutual_information` y una
    `drop_candidate` booleana para las variables en el cuartil inferior de MI.
    """
    if task not in {"regression", "classification"}:
        raise ValueError("task debe ser 'regression' o 'classification'.")

    if task == "regression":
        y_discrete = pd.qcut(y, q=quantile_bins, duplicates="drop").astype(str)
    else:
        y_discrete = y.astype(str)

    rows: list[dict[str, object]] = []
    for column in X.columns:
        series = X[column]
        if pd.api.types.is_numeric_dtype(series):
            correlation = _spearman(series, y)
            rows.append(
                {
                    "feature": column,
                    "type": "numeric",
                    "correlation": correlation,
                    "correlation_kind": "spearman",
                }
            )
        else:
            correlation = _cramers_v(series.astype(object), y_discrete)
            rows.append(
                {
                    "feature": column,
                    "type": "categorical",
                    "correlation": correlation,
                    "correlation_kind": "cramers_v",
                }
            )
    table = pd.DataFrame(rows)
    mi = _mutual_information(X, y, task, random_state)
    table["mutual_information"] = table["feature"].map(mi)
    table["abs_correlation"] = table["correlation"].abs()

    threshold = table["mutual_information"].quantile(0.25)
    table["drop_candidate"] = table["mutual_information"] <= threshold

    ordered = table.sort_values("mutual_information", ascending=False).reset_index(drop=True)
    return ordered[
        [
            "feature",
            "type",
            "correlation",
            "correlation_kind",
            "abs_correlation",
            "mutual_information",
            "drop_candidate",
        ]
    ]
