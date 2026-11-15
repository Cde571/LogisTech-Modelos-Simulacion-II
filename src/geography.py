"""Enriquecimiento geográfico del dataset a nivel de pedido.

El dataset Olist incluye una tabla de geolocalización (`olist_geolocation_dataset`)
que mapea prefijos de código postal (CEP) a coordenadas (latitud, longitud).
Al combinarlo con los códigos postales del cliente (en `olist_customers_dataset`)
y del vendedor (en `olist_sellers_dataset`) podemos estimar la distancia
cliente ↔ vendedor por pedido y exponerla como predictor adicional.

El supuesto operativo es que la distancia física es un factor estructural del
tiempo de entrega en un país grande como Brasil. Esta señal está disponible
en el momento de la compra, así que no introduce fuga de información.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: pd.Series | np.ndarray | float,
    lon1: pd.Series | np.ndarray | float,
    lat2: pd.Series | np.ndarray | float,
    lon2: pd.Series | np.ndarray | float,
) -> np.ndarray:
    """Distancia ortodrómica en kilómetros entre pares de coordenadas."""
    lat1_r, lon1_r, lat2_r, lon2_r = (
        np.radians(np.asarray(lat1, dtype=float)),
        np.radians(np.asarray(lon1, dtype=float)),
        np.radians(np.asarray(lat2, dtype=float)),
        np.radians(np.asarray(lon2, dtype=float)),
    )
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    inner = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def aggregate_geolocation(geolocation: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una coordenada representativa por prefijo de CEP.

    La tabla original tiene varias filas por prefijo porque incluye cada
    dirección geocodificada. Usamos la mediana para reducir sensibilidad a
    outliers de geocodificación.
    """
    required = {"geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"}
    missing = required.difference(geolocation.columns)
    if missing:
        raise KeyError(f"La tabla de geolocalización no tiene columnas: {sorted(missing)}")

    aggregated = (
        geolocation.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(lat=("geolocation_lat", "median"), lon=("geolocation_lng", "median"))
    )
    return aggregated.rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix"})


def _order_seller_distances(
    items: pd.DataFrame, sellers: pd.DataFrame, zip_coords: pd.DataFrame
) -> pd.DataFrame:
    """Devuelve una fila por par (order_id, seller_id) con la coordenada del vendedor."""
    seller_coords = sellers.merge(
        zip_coords, left_on="seller_zip_code_prefix", right_on="zip_code_prefix", how="left"
    )[["seller_id", "lat", "lon"]].rename(columns={"lat": "seller_lat", "lon": "seller_lon"})

    order_sellers = (
        items[["order_id", "seller_id"]].drop_duplicates().merge(
            seller_coords, on="seller_id", how="left"
        )
    )
    return order_sellers


def attach_order_geography(
    dataset: pd.DataFrame, tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Añade columnas de distancia cliente↔vendedor al dataset por pedido.

    El dataset de entrada se espera con una fila por `order_id` e incluir la
    columna `customer_zip_code_prefix`. Se devuelven nuevas columnas:

    - ``customer_lat`` / ``customer_lon``
    - ``mean_customer_seller_distance_km``
    - ``max_customer_seller_distance_km``
    - ``sellers_with_coordinates``

    Los pedidos cuyo prefijo postal no tiene coordenada quedan con NaN y se
    tratarán como faltantes en el pipeline. No se descartan filas.
    """
    required_tables = {"geolocation", "customers", "items", "sellers"}
    missing = required_tables.difference(tables)
    if missing:
        raise KeyError(f"Faltan tablas para el enriquecimiento geográfico: {sorted(missing)}")
    if "customer_zip_code_prefix" not in dataset.columns:
        raise KeyError(
            "El dataset no contiene 'customer_zip_code_prefix'; "
            "cargue primero `build_order_level_dataset`."
        )

    zip_coords = aggregate_geolocation(tables["geolocation"])

    customer_coords = (
        dataset[["order_id", "customer_zip_code_prefix"]]
        .merge(zip_coords, left_on="customer_zip_code_prefix", right_on="zip_code_prefix", how="left")
        .rename(columns={"lat": "customer_lat", "lon": "customer_lon"})
        [["order_id", "customer_lat", "customer_lon"]]
    )

    order_sellers = _order_seller_distances(tables["items"], tables["sellers"], zip_coords)
    order_sellers = order_sellers.merge(customer_coords, on="order_id", how="left")
    order_sellers["distance_km"] = haversine_km(
        order_sellers["customer_lat"],
        order_sellers["customer_lon"],
        order_sellers["seller_lat"],
        order_sellers["seller_lon"],
    )

    distances = order_sellers.groupby("order_id", as_index=False).agg(
        mean_customer_seller_distance_km=("distance_km", "mean"),
        max_customer_seller_distance_km=("distance_km", "max"),
        sellers_with_coordinates=("distance_km", lambda s: int(s.notna().sum())),
    )

    enriched = dataset.merge(customer_coords, on="order_id", how="left", validate="one_to_one")
    enriched = enriched.merge(distances, on="order_id", how="left", validate="one_to_one")

    if enriched["order_id"].duplicated().any():
        raise ValueError("El enriquecimiento geográfico rompió la unicidad por order_id.")
    return enriched
