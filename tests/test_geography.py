import unittest

import numpy as np
import pandas as pd

from src.geography import aggregate_geolocation, attach_order_geography, haversine_km


class GeographyTests(unittest.TestCase):
    def test_haversine_sao_paulo_rio_matches_reference(self):
        # SP y RJ: la ortodrómica real ronda los 360 km.
        distance = haversine_km(-23.55, -46.63, -22.90, -43.21)
        self.assertAlmostEqual(float(distance), 360.0, delta=10.0)

    def test_haversine_zero_distance_when_coordinates_match(self):
        self.assertAlmostEqual(float(haversine_km(10.0, 20.0, 10.0, 20.0)), 0.0, places=6)

    def test_haversine_supports_array_inputs(self):
        lat1 = np.array([-23.55, 0.0])
        lon1 = np.array([-46.63, 0.0])
        lat2 = np.array([-22.90, 0.0])
        lon2 = np.array([-43.21, 0.0])
        distances = haversine_km(lat1, lon1, lat2, lon2)
        self.assertEqual(distances.shape, (2,))
        self.assertAlmostEqual(float(distances[1]), 0.0, places=6)

    def test_aggregate_geolocation_uses_median_per_prefix(self):
        geolocation = pd.DataFrame(
            {
                "geolocation_zip_code_prefix": ["01000", "01000", "02000"],
                "geolocation_lat": [-23.5, -23.7, -22.9],
                "geolocation_lng": [-46.6, -46.8, -43.2],
            }
        )
        aggregated = aggregate_geolocation(geolocation)
        self.assertSetEqual(set(aggregated.columns), {"zip_code_prefix", "lat", "lon"})
        self.assertEqual(len(aggregated), 2)
        row = aggregated.loc[aggregated["zip_code_prefix"] == "01000"].iloc[0]
        self.assertAlmostEqual(row["lat"], -23.6)
        self.assertAlmostEqual(row["lon"], -46.7)

    def test_attach_order_geography_adds_distance_columns(self):
        dataset = pd.DataFrame(
            {
                "order_id": ["o1", "o2"],
                "customer_zip_code_prefix": ["01000", "02000"],
            }
        )
        items = pd.DataFrame(
            {
                "order_id": ["o1", "o1", "o2"],
                "seller_id": ["s1", "s2", "s1"],
            }
        )
        sellers = pd.DataFrame(
            {
                "seller_id": ["s1", "s2"],
                "seller_zip_code_prefix": ["02000", "01000"],
            }
        )
        geolocation = pd.DataFrame(
            {
                "geolocation_zip_code_prefix": ["01000", "02000"],
                "geolocation_lat": [-23.55, -22.90],
                "geolocation_lng": [-46.63, -43.21],
            }
        )
        tables = {
            "geolocation": geolocation,
            "customers": pd.DataFrame(),
            "items": items,
            "sellers": sellers,
        }
        enriched = attach_order_geography(dataset, tables)
        self.assertIn("mean_customer_seller_distance_km", enriched.columns)
        self.assertIn("max_customer_seller_distance_km", enriched.columns)
        self.assertIn("sellers_with_coordinates", enriched.columns)
        self.assertEqual(len(enriched), 2)

        # o2 tiene un único vendedor (s1 en 02000) y el cliente está en 02000:
        # misma coordenada, distancia 0.
        o2 = enriched.loc[enriched["order_id"] == "o2"].iloc[0]
        self.assertAlmostEqual(o2["mean_customer_seller_distance_km"], 0.0, places=3)
        self.assertEqual(o2["sellers_with_coordinates"], 1)

        # o1 tiene dos vendedores, uno en SP y otro en RJ.
        o1 = enriched.loc[enriched["order_id"] == "o1"].iloc[0]
        self.assertGreater(o1["mean_customer_seller_distance_km"], 100.0)
        self.assertEqual(o1["sellers_with_coordinates"], 2)

    def test_attach_order_geography_preserves_order_uniqueness(self):
        dataset = pd.DataFrame(
            {
                "order_id": ["o1"],
                "customer_zip_code_prefix": ["01000"],
            }
        )
        tables = {
            "geolocation": pd.DataFrame(
                {
                    "geolocation_zip_code_prefix": ["01000"],
                    "geolocation_lat": [-23.55],
                    "geolocation_lng": [-46.63],
                }
            ),
            "customers": pd.DataFrame(),
            "items": pd.DataFrame({"order_id": ["o1"], "seller_id": ["s1"]}),
            "sellers": pd.DataFrame(
                {"seller_id": ["s1"], "seller_zip_code_prefix": ["01000"]}
            ),
        }
        enriched = attach_order_geography(dataset, tables)
        self.assertTrue(enriched["order_id"].is_unique)


if __name__ == "__main__":
    unittest.main()
