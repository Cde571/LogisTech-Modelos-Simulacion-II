"""Experimento baseline del Entregable II: predictores pre-compra puros.

Reproduce el pipeline del notebook 03_experimentos.ipynb usando unicamente las
columnas disponibles en el momento de la compra (sin distancia geografica ni
historicos del vendedor). Los resultados se guardan en
``outputs/experimentos/01_baseline_pre_compra/`` y documentan el punto de
partida frente al que se mediran las mejoras posteriores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments import run_task
from src.models import ExperimentConfig
from src.preprocessing import build_order_level_dataset, load_olist_tables, modeling_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=15_000, help="None = todas las filas (usar -1)")
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--output", default="outputs/experimentos/01_baseline_pre_compra")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_rows = None if args.max_rows is None or args.max_rows < 0 else args.max_rows
    config = ExperimentConfig(
        max_rows=max_rows,
        cv_splits=args.cv_splits,
        bootstrap_iterations=args.bootstrap,
    )
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{perf_counter():.0f}s] Config: {config}")

    tables = load_olist_tables(ROOT / "data" / "raw")
    orders = build_order_level_dataset(tables)
    assert orders["order_id"].is_unique
    print(
        f"[{perf_counter():.0f}s] Pedidos agregados: {len(orders):,} "
        f"periodo {orders.order_purchase_timestamp.min()} a {orders.order_purchase_timestamp.max()}"
    )

    results: dict[str, dict] = {}
    for task, target in [("regression", "delivery_time_days"), ("classification", "late_delivery")]:
        X, y, dates = modeling_frame(orders, target)
        print(f"[{perf_counter():.0f}s] ---- {task.upper()} target={target} rows={len(X):,}")
        task_result = run_task(task, X, y, dates, config)
        for name in ["cv_results", "ranking", "feature_importance", "reductions"]:
            task_result[name].to_csv(output_dir / f"{task}_{name}.csv", index=False)
        champion = task_result["ranking"].iloc[0]["model"]
        primary = task_result["ranking"].iloc[0]["test_primary"]
        ci_low = task_result["ranking"].iloc[0]["test_ci95_low"]
        ci_high = task_result["ranking"].iloc[0]["test_ci95_high"]
        print(
            f"[{perf_counter():.0f}s] {task} champion={champion} "
            f"test_primary={primary:.4f} [{ci_low:.4f}, {ci_high:.4f}]"
        )
        results[task] = {
            "champion": champion,
            "test_primary": float(primary),
            "test_ci95": [float(ci_low), float(ci_high)],
        }

    summary = {
        "sample_rows": max_rows,
        "cv_splits": config.cv_splits,
        "bootstrap_iterations": config.bootstrap_iterations,
        **results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[{perf_counter():.0f}s] Resultados en {output_dir}")


if __name__ == "__main__":
    main()
