"""Smoke-test end-to-end de `run_task` con datos sinteticos.

Se ejecuta explicitamente con `SMOKE=1 python -m unittest tests.test_smoke`
para no afectar el flujo rapido de pruebas unitarias. Verifica que la cadena
preprocess -> tune -> fit -> permutation -> PCA/UMAP corre sin errores sobre
un conjunto reducido que replica la estructura categorica de Olist.
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import pandas as pd

from src.experiments import run_task
from src.models import ExperimentConfig


@unittest.skipUnless(os.getenv("SMOKE"), "define SMOKE=1 para ejecutar el smoke test")
class RunTaskSmokeTests(unittest.TestCase):
    def _synthetic_dataset(self, task: str, n: int = 600) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            {
                "payment_type": rng.choice(
                    ["credit", "boleto", "voucher", "debit"], size=n
                ),
                "customer_state": rng.choice(
                    [f"S{index:02d}" for index in range(27)], size=n
                ),
                "product_category_name": rng.choice(
                    [f"cat_{index}" for index in range(60)], size=n
                ),
                "customer_city": rng.choice(
                    [f"city_{index}" for index in range(200)], size=n
                ),
                "payment_value": rng.normal(100, 30, size=n),
                "estimated_delivery_days": rng.integers(5, 20, size=n).astype(float),
                "total_freight_value": rng.normal(20, 5, size=n),
                "item_count": rng.integers(1, 4, size=n).astype(float),
            }
        )
        if task == "regression":
            y = pd.Series(
                X["estimated_delivery_days"]
                + rng.normal(0, 3, size=n)
                + 0.05 * X["total_freight_value"],
                name="delivery_time_days",
            )
        else:
            logits = 0.1 * (X["estimated_delivery_days"] - 12) + rng.normal(0, 1, size=n)
            y = pd.Series((logits > 0).astype(int), name="late_delivery")
        dates = pd.Series(pd.date_range("2020-01-01", periods=n, freq="h"))
        return X, y, dates

    def test_run_task_regression_pipeline_completes(self):
        X, y, dates = self._synthetic_dataset("regression")
        config = ExperimentConfig(
            max_rows=None,
            cv_splits=3,
            bootstrap_iterations=50,
            low_cardinality_max=30,
            high_cardinality_min=100,
        )
        result = run_task("regression", X, y, dates, config)
        self.assertIn("ranking", result)
        self.assertFalse(result["ranking"].empty)
        self.assertIn("test_MAE", result["ranking"].columns)
        self.assertIn("reductions", result)
        self.assertFalse(result["reductions"].empty)

    def test_run_task_classification_pipeline_completes(self):
        X, y, dates = self._synthetic_dataset("classification")
        config = ExperimentConfig(
            max_rows=None,
            cv_splits=3,
            bootstrap_iterations=50,
            low_cardinality_max=30,
            high_cardinality_min=100,
        )
        result = run_task("classification", X, y, dates, config)
        self.assertIn("ranking", result)
        self.assertFalse(result["ranking"].empty)
        self.assertIn("test_ROC_AUC", result["ranking"].columns)


if __name__ == "__main__":
    unittest.main()
