import unittest

import pandas as pd

from src.entregable_ii import split_train_validation_test, temporal_sample
from src.preprocessing import LEAKAGE_COLUMNS, assert_no_leakage, predictor_columns


class PipelineIntegrityTests(unittest.TestCase):
    def test_predictors_exclude_every_leakage_column(self):
        frame = pd.DataFrame({name: [0] for name in LEAKAGE_COLUMNS})
        frame["customer_state"] = ["SP"]
        selected = predictor_columns(frame)
        self.assertEqual(selected, ["customer_state"])
        self.assertTrue(LEAKAGE_COLUMNS.isdisjoint(selected))

    def test_leakage_guard_rejects_future_signal(self):
        with self.assertRaises(ValueError):
            assert_no_leakage(["customer_state", "order_delivered_customer_date"])

    def test_temporal_sample_preserves_endpoints_and_order(self):
        X = pd.DataFrame({"x": range(100)})
        y = pd.Series(range(100))
        dates = pd.Series(pd.date_range("2020-01-01", periods=100))
        sampled_X, sampled_y, sampled_dates = temporal_sample(X, y, dates, 10)
        self.assertEqual(sampled_X.iloc[0, 0], 0)
        self.assertEqual(sampled_X.iloc[-1, 0], 99)
        self.assertTrue(sampled_dates.is_monotonic_increasing)
        self.assertEqual(sampled_y.tolist(), sampled_X["x"].tolist())

    def test_split_is_strictly_chronological(self):
        X = pd.DataFrame({"x": range(1_000)})
        y = pd.Series(range(1_000))
        dates = pd.Series(pd.date_range("2020-01-01", periods=1_000))
        parts = split_train_validation_test(X, y, dates)
        self.assertLess(parts["d_train"].max(), parts["d_validation"].min())
        self.assertLess(parts["d_validation"].max(), parts["d_test"].min())
        self.assertEqual(
            [len(parts["X_train"]), len(parts["X_validation"]), len(parts["X_test"])],
            [600, 200, 200],
        )


if __name__ == "__main__":
    unittest.main()
