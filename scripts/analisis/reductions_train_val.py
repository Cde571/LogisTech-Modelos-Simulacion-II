"""Regenera los CSVs de reducción dimensional con métricas en train/val/test.

Reutiliza el ranking y los hiperparámetros ya seleccionados por el
experimento 02 para evitar re-ejecutar la búsqueda completa de CV, pero
vuelve a ajustar los dos mejores modelos sobre las representaciones PCA
y UMAP y guarda las métricas de los tres splits.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation import reduction_comparison
from src.geography import attach_order_geography
from src.models import ExperimentConfig, model_catalog
from src.preprocessing import build_order_level_dataset, load_olist_tables, modeling_frame
from src.splits import split_train_validation_test, temporal_sample

EXPERIMENTO = ROOT / "outputs" / "experimentos" / "02_distancia_geografica"


def _best_from_ranking(ranking: pd.DataFrame, task: str, random_state: int) -> dict[str, dict]:
    catalogo = model_catalog(task, random_state)
    best: dict[str, dict] = {}
    for _, row in ranking.iterrows():
        family = row["model"]
        if family not in catalogo:
            continue
        estimator, _ = catalogo[family]
        best[family] = {
            "estimator": estimator,
            "params": ast.literal_eval(row["best_params"]),
            "cv_mean": row.get("cv_mean"),
            "cv_worst": row.get("cv_worst"),
        }
    return best


def main() -> None:
    config = ExperimentConfig(max_rows=None)
    tables = load_olist_tables(ROOT / "data" / "raw")
    orders = build_order_level_dataset(tables)
    orders = attach_order_geography(orders, tables)

    for task, target in [("regression", "delivery_time_days"), ("classification", "late_delivery")]:
        print(f"[{task}] cargando datos")
        X, y, dates = modeling_frame(orders, target)
        X, y, dates = temporal_sample(X, y, dates, config.max_rows)
        parts = split_train_validation_test(X, y, dates)

        ranking = pd.read_csv(EXPERIMENTO / f"{task}_ranking.csv")
        best = _best_from_ranking(ranking, task, config.random_state)
        print(f"[{task}] reducciones PCA y UMAP sobre {len(parts['X_train']):,} train / "
              f"{len(parts['X_validation']):,} val / {len(parts['X_test']):,} test")

        reductions = reduction_comparison(task, parts, ranking, best, config)
        destino = EXPERIMENTO / f"{task}_reductions.csv"
        reductions.to_csv(destino, index=False)
        print(f"[{task}] escrito {destino}")


if __name__ == "__main__":
    main()
