"""Preparación reproducible de Olist a nivel de pedido.

La predicción se sitúa en el instante de compra. Las columnas posteriores a
ese instante se usan únicamente para construir los objetivos y nunca como
predictores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

REQUIRED_TABLES = ("orders", "items", "customers", "products", "payments")

DATE_COLUMNS = (
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
)

# Nunca deben entrar a X para una predicción realizada al comprar.
LEAKAGE_COLUMNS = {
    "order_status",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "review_score",
    "review_comment_title",
    "review_comment_message",
    "review_creation_date",
    "review_answer_timestamp",
    "delivery_time_days",
    "late_delivery",
}


def load_olist_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Carga las tablas conocidas y falla con un mensaje claro si falta alguna requerida."""
    root = Path(data_dir)
    missing = [FILES[name] for name in REQUIRED_TABLES if not (root / FILES[name]).exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos requeridos en "
            f"{root.resolve()}: {', '.join(missing)}. "
            "Consulte data/README.md para descargarlos."
        )

    tables: dict[str, pd.DataFrame] = {}
    for name, filename in FILES.items():
        path = root / filename
        if path.exists():
            tables[name] = pd.read_csv(path)
    return tables


def convert_order_dates(orders: pd.DataFrame) -> pd.DataFrame:
    """Convierte a datetime las fechas de pedidos sin modificar el DataFrame original."""
    result = orders.copy()
    for column in DATE_COLUMNS:
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def add_targets(orders: pd.DataFrame) -> pd.DataFrame:
    """Construye los objetivos solo cuando existen las fechas necesarias."""
    result = convert_order_dates(orders)
    delivered = result["order_delivered_customer_date"]
    purchased = result["order_purchase_timestamp"]
    estimated = result["order_estimated_delivery_date"]

    result["delivery_time_days"] = (delivered - purchased).dt.total_seconds() / 86_400
    result["late_delivery"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    observed = delivered.notna() & estimated.notna()
    result.loc[observed, "late_delivery"] = (
        delivered.loc[observed] > estimated.loc[observed]
    ).astype("int64")

    # Duraciones negativas señalan registros inválidos y no se usan como targets.
    result.loc[result["delivery_time_days"] < 0, "delivery_time_days"] = np.nan
    return result


def _mode_or_missing(series: pd.Series) -> object:
    modes = series.dropna().mode()
    return modes.iloc[0] if not modes.empty else pd.NA


def _aggregate_items(items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    enriched = items.merge(products, on="product_id", how="left", validate="many_to_one")
    aggregations = {
        "item_count": ("order_item_id", "count"),
        "seller_count": ("seller_id", "nunique"),
        "total_price": ("price", "sum"),
        "total_freight_value": ("freight_value", "sum"),
        "product_category_name": ("product_category_name", _mode_or_missing),
        "mean_product_weight_g": ("product_weight_g", "mean"),
        "mean_product_length_cm": ("product_length_cm", "mean"),
        "mean_product_height_cm": ("product_height_cm", "mean"),
        "mean_product_width_cm": ("product_width_cm", "mean"),
    }
    return enriched.groupby("order_id", as_index=False).agg(**aggregations)


def _aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    return payments.groupby("order_id", as_index=False).agg(
        payment_type=("payment_type", _mode_or_missing),
        payment_installments=("payment_installments", "max"),
        payment_value=("payment_value", "sum"),
        payment_records=("payment_sequential", "count"),
    )


def build_order_level_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une Olist a una fila por pedido y agrega variables disponibles al comprar."""
    missing = [name for name in REQUIRED_TABLES if name not in tables]
    if missing:
        raise KeyError(f"Faltan tablas requeridas: {missing}")

    orders = add_targets(tables["orders"])
    item_features = _aggregate_items(tables["items"], tables["products"])
    payment_features = _aggregate_payments(tables["payments"])

    dataset = (
        orders.merge(
            tables["customers"], on="customer_id", how="left", validate="many_to_one"
        )
        .merge(item_features, on="order_id", how="left", validate="one_to_one")
        .merge(payment_features, on="order_id", how="left", validate="one_to_one")
    )

    purchased = dataset["order_purchase_timestamp"]
    dataset["purchase_year"] = purchased.dt.year
    dataset["purchase_month"] = purchased.dt.month
    dataset["purchase_dayofweek"] = purchased.dt.dayofweek
    dataset["purchase_hour"] = purchased.dt.hour
    dataset["estimated_delivery_days"] = (
        dataset["order_estimated_delivery_date"] - purchased
    ).dt.total_seconds() / 86_400

    if dataset["order_id"].duplicated().any():
        raise ValueError("La agregación no produjo una fila única por order_id.")
    return dataset


def predictor_columns(dataset: pd.DataFrame) -> list[str]:
    """Devuelve el conjunto inicial de variables conocidas al comprar."""
    candidates = [
        "customer_city",
        "customer_state",
        "product_category_name",
        "payment_type",
        "payment_installments",
        "payment_value",
        "payment_records",
        "item_count",
        "seller_count",
        "total_price",
        "total_freight_value",
        "mean_product_weight_g",
        "mean_product_length_cm",
        "mean_product_height_cm",
        "mean_product_width_cm",
        "purchase_year",
        "purchase_month",
        "purchase_dayofweek",
        "purchase_hour",
        "estimated_delivery_days",
    ]
    selected = [column for column in candidates if column in dataset]
    assert_no_leakage(selected)
    return selected


def assert_no_leakage(columns: Iterable[str]) -> None:
    """Falla si se intenta usar una señal futura o un target como predictor."""
    found = LEAKAGE_COLUMNS.intersection(columns)
    if found:
        raise ValueError(f"Variables con fuga de información detectadas: {sorted(found)}")


def modeling_frame(
    dataset: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Crea X, y y la fecha de compra para partición cronológica."""
    if target not in {"delivery_time_days", "late_delivery"}:
        raise ValueError("target debe ser 'delivery_time_days' o 'late_delivery'.")
    columns = predictor_columns(dataset)
    valid = dataset[target].notna() & dataset["order_purchase_timestamp"].notna()
    frame = dataset.loc[valid].sort_values("order_purchase_timestamp")
    X = frame[columns].copy()
    # Algunas combinaciones de pandas/scikit-learn (incluido Colab) no pueden
    # evaluar pd.NA dentro de SimpleImputer. Normalizar a np.nan conserva el
    # significado y evita el error "boolean value of NA is ambiguous".
    categorical = X.select_dtypes(include=["object", "category", "string"]).columns
    for column in categorical:
        X[column] = X[column].astype(object).where(X[column].notna(), np.nan)
    y = frame[target].astype("int64" if target == "late_delivery" else "float64")
    dates = frame["order_purchase_timestamp"].copy()
    return X, y, dates
