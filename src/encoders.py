"""Codificacion diferenciada por cardinalidad declarada en el reporte.

El preprocesador unico que consume el pipeline enruta cada columna categorica
segun su cardinalidad observada en el conjunto de ajuste:

    - cardinalidad <= `low_cardinality_max`
      (payment_type, customer_state en Olist): one-hot sin truncar.
    - cardinalidad entre los dos umbrales
      (product_category_name): one-hot con agrupamiento `min_frequency` que
      colapsa en una categoria sintetica las clases infrecuentes.
    - cardinalidad >= `high_cardinality_min`
      (customer_city): codificacion de frecuencia (conteo observado) seguida
      por estandarizacion z para no sesgar modelos lineales/SVM. Las
      categorias no vistas valen 0, lo que se interpreta como ciudades con
      densidad logistica historica desconocida.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models import ExperimentConfig


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Reemplaza cada categoria por su conteo en entrenamiento."""

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=object)
        self.frequencies_ = []
        for column_index in range(array.shape[1]):
            values, counts = np.unique(array[:, column_index], return_counts=True)
            self.frequencies_.append(
                {value: int(count) for value, count in zip(values, counts)}
            )
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X):
        array = np.asarray(X, dtype=object)
        output = np.zeros(array.shape, dtype=float)
        for column_index in range(array.shape[1]):
            mapping = self.frequencies_[column_index]
            output[:, column_index] = [mapping.get(value, 0) for value in array[:, column_index]]
        return output

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{index}" for index in range(self.n_features_in_)]
        return np.asarray([f"{name}_frequency" for name in input_features], dtype=object)


def _classify_categoricals(
    X: pd.DataFrame, categorical: list[str], config: ExperimentConfig
) -> tuple[list[str], list[str], list[str]]:
    low: list[str] = []
    mid: list[str] = []
    high: list[str] = []
    for column in categorical:
        cardinality = int(X[column].nunique(dropna=True))
        if cardinality >= config.high_cardinality_min:
            high.append(column)
        elif cardinality > config.low_cardinality_max:
            mid.append(column)
        else:
            low.append(column)
    return low, mid, high


def make_preprocessor(X: pd.DataFrame, config: ExperimentConfig) -> ColumnTransformer:
    """ColumnTransformer con codificacion diferenciada por cardinalidad."""
    categorical = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric = X.columns.difference(categorical).tolist()
    low_cardinality, mid_cardinality, high_cardinality = _classify_categoricals(
        X, categorical, config
    )

    transformers: list[tuple[str, Any, list[str]]] = [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric,
        )
    ]
    if low_cardinality:
        transformers.append(
            (
                "categorical_low",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                low_cardinality,
            )
        )
    if mid_cardinality:
        transformers.append(
            (
                "categorical_mid",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=config.min_frequency,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                mid_cardinality,
            )
        )
    if high_cardinality:
        transformers.append(
            (
                "categorical_high",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("frequency", FrequencyEncoder()),
                        ("scaler", StandardScaler()),
                    ]
                ),
                high_cardinality,
            )
        )
    return ColumnTransformer(transformers, verbose_feature_names_out=False)
