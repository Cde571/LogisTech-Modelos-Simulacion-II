"""Comprobaciones sobre el análisis individual de variables."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.variable_analysis import variable_analysis


class VariableAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        n = 400
        signal = rng.normal(size=n)
        noise = rng.normal(size=n)
        category = np.where(signal > 0, "alta", "baja")
        y = (signal > 0).astype(int)
        self.X = pd.DataFrame(
            {
                "senal_numerica": signal,
                "ruido_numerico": noise,
                "categoria_fuerte": category,
                "categoria_ruido": rng.choice(["a", "b", "c"], size=n),
            }
        )
        self.y_clf = pd.Series(y)
        self.y_reg = pd.Series(signal + 0.1 * noise)

    def test_strong_variable_tops_the_classification_ranking(self) -> None:
        analysis = variable_analysis(self.X, self.y_clf, task="classification")
        top_feature = analysis.iloc[0]["feature"]
        self.assertIn(top_feature, {"senal_numerica", "categoria_fuerte"})
        noise_row = analysis[analysis["feature"] == "ruido_numerico"].iloc[0]
        signal_row = analysis[analysis["feature"] == "senal_numerica"].iloc[0]
        self.assertGreater(
            signal_row["mutual_information"],
            noise_row["mutual_information"],
        )

    def test_noise_variable_ranks_below_signal(self) -> None:
        analysis = variable_analysis(self.X, self.y_clf, task="classification")
        positions = {row["feature"]: idx for idx, row in analysis.iterrows()}
        self.assertLess(positions["senal_numerica"], positions["ruido_numerico"])
        self.assertLess(positions["categoria_fuerte"], positions["categoria_ruido"])
        self.assertTrue(analysis["drop_candidate"].any())

    def test_correlation_kind_matches_variable_type(self) -> None:
        analysis = variable_analysis(self.X, self.y_reg, task="regression")
        row_numeric = analysis[analysis["feature"] == "senal_numerica"].iloc[0]
        row_categorical = analysis[analysis["feature"] == "categoria_fuerte"].iloc[0]
        self.assertEqual(row_numeric["correlation_kind"], "spearman")
        self.assertEqual(row_categorical["correlation_kind"], "cramers_v")


if __name__ == "__main__":
    unittest.main()
