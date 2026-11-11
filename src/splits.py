"""Muestreo temporal y particion cronologica 60/20/20.

El muestreo preserva el orden y los extremos del periodo para no sesgar la
evaluacion. La particion separa estrictamente los pedidos del pasado para
entrenar, los intermedios para validar hiperparametros y los mas recientes
para evaluar generalizacion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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
    return {
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
