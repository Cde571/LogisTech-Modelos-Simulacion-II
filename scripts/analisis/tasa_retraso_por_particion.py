"""Reporta la tasa de retraso en cada partición temporal.

Deja escrita en `outputs/entregable_ii/shift_por_particion.json` la
fracción de entregas tardías observada en train, validación y prueba,
junto con el rango de fechas que las define. Es la evidencia primaria
del *shift* temporal que se discute en el Entregable II.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.geography import attach_order_geography
from src.models import ExperimentConfig
from src.preprocessing import build_order_level_dataset, load_olist_tables, modeling_frame
from src.splits import split_train_validation_test, temporal_sample


def describe_split(part: str, parts: dict, target: str) -> dict:
    dates = parts[f"d_{part}"]
    y = parts[f"y_{part}"]
    return {
        "n": int(len(y)),
        "late_rate": float(y.mean()) if target == "late_delivery" else None,
        "mean_days": float(y.mean()) if target == "delivery_time_days" else None,
        "fecha_min": str(dates.min()),
        "fecha_max": str(dates.max()),
    }


def main() -> None:
    config = ExperimentConfig(max_rows=None)
    tables = load_olist_tables(ROOT / "data" / "raw")
    orders = build_order_level_dataset(tables)
    orders = attach_order_geography(orders, tables)

    summary: dict[str, dict] = {}
    for task, target in [
        ("classification", "late_delivery"),
        ("regression", "delivery_time_days"),
    ]:
        X, y, dates = modeling_frame(orders, target)
        X, y, dates = temporal_sample(X, y, dates, config.max_rows)
        parts = split_train_validation_test(X, y, dates)
        summary[task] = {
            split: describe_split(split, parts, target)
            for split in ("train", "validation", "test")
        }

    out = ROOT / "outputs" / "entregable_ii" / "shift_por_particion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado: {out}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
