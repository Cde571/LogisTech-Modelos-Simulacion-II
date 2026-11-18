"""Genera las figuras del Entregable II a partir de los CSVs existentes.

Produce cuatro PDFs vectoriales en report/figures/:
    mae_ic95.pdf             — MAE sobre prueba con IC95 por familia (regresión).
    roc_auc_ic95.pdf         — ROC-AUC sobre prueba con IC95 por familia (clasificación).
    permutation_importance.pdf — Top 10 de importancia por permutación en validación.
    umap_scatter.pdf         — Proyección UMAP 2D del conjunto de validación coloreada por retraso.

Las tres primeras leen únicamente los CSV del experimento 02 (la configuración
principal del informe). La cuarta recomputa la proyección 2D porque los CSV
solo conservan el resumen numérico; el ajuste es no supervisado.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXPERIMENTO = ROOT / "outputs" / "experimentos" / "02_distancia_geografica"
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def _etiqueta_modelo(nombre: str) -> str:
    return nombre.replace(" (SVM)", "").replace(" (ensamble)", "").replace(" (red neuronal)", "").replace(" (no parametrico)", "").replace(" (parametrico)", "").replace("Baseline trivial (mediana)", "Baseline (mediana)").replace("Baseline trivial (clase mayoritaria)", "Baseline (clase mayor.)")


def figura_mae() -> None:
    ranking = pd.read_csv(EXPERIMENTO / "regression_ranking.csv")
    ranking = ranking.sort_values("test_primary", ascending=False)
    etiquetas = [_etiqueta_modelo(m) for m in ranking["model"]]
    centro = ranking["test_primary"].to_numpy()
    low = (centro - ranking["test_ci95_low"]).to_numpy()
    high = (ranking["test_ci95_high"] - centro).to_numpy()
    baseline_mediana = float(ranking.loc[ranking["model"].str.startswith("Baseline"), "test_primary"].iloc[0])

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    colores = ["#8c8c8c" if m.startswith("Baseline") else "#1f4e79" for m in ranking["model"]]
    ax.barh(etiquetas, centro, xerr=[low, high], color=colores, edgecolor="black", linewidth=0.5, capsize=3)
    ax.axvline(baseline_mediana, color="#8c8c8c", linestyle="--", linewidth=0.8)
    ax.set_xlabel("MAE sobre prueba (días)")
    ax.set_xlim(0, max(ranking["test_ci95_high"]) * 1.08)
    fig.savefig(FIG_DIR / "mae_ic95.pdf")
    plt.close(fig)


def figura_roc_auc() -> None:
    ranking = pd.read_csv(EXPERIMENTO / "classification_ranking.csv")
    ranking = ranking.sort_values("test_primary", ascending=True)
    etiquetas = [_etiqueta_modelo(m) for m in ranking["model"]]
    centro = ranking["test_primary"].to_numpy()
    low = (centro - ranking["test_ci95_low"]).to_numpy()
    high = (ranking["test_ci95_high"] - centro).to_numpy()

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    colores = ["#8c8c8c" if m.startswith("Baseline") else "#1f4e79" for m in ranking["model"]]
    ax.barh(etiquetas, centro, xerr=[low, high], color=colores, edgecolor="black", linewidth=0.5, capsize=3)
    ax.axvline(0.5, color="#8c8c8c", linestyle="--", linewidth=0.8)
    ax.set_xlabel("ROC--AUC sobre prueba")
    ax.set_xlim(0.45, max(ranking["test_ci95_high"]) * 1.05)
    fig.savefig(FIG_DIR / "roc_auc_ic95.pdf")
    plt.close(fig)


def figura_importancia() -> None:
    importancias = pd.read_csv(EXPERIMENTO / "regression_feature_importance.csv")
    top = importancias.head(10).iloc[::-1]
    etiquetas = top["feature"].str.replace("_", r"\_", regex=False)

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.barh(
        etiquetas,
        top["importance_mean"],
        xerr=top["importance_std"],
        color="#1f4e79",
        edgecolor="black",
        linewidth=0.5,
        capsize=2,
    )
    ax.set_xlabel("Caída media de $R^2$ en validación")
    fig.savefig(FIG_DIR / "permutation_importance.pdf")
    plt.close(fig)


def figura_umap_scatter() -> None:
    from umap import UMAP

    from src.encoders import make_preprocessor
    from src.geography import attach_order_geography
    from src.models import ExperimentConfig
    from src.preprocessing import build_order_level_dataset, load_olist_tables, modeling_frame
    from src.splits import split_train_validation_test

    config = ExperimentConfig(max_rows=None)
    tables = load_olist_tables(ROOT / "data" / "raw")
    orders = build_order_level_dataset(tables)
    orders = attach_order_geography(orders, tables)
    X, y, dates = modeling_frame(orders, "late_delivery")
    parts = split_train_validation_test(X, y, dates)

    preprocesador = make_preprocessor(parts["X_train"], config)
    X_train_mat = preprocesador.fit_transform(parts["X_train"])
    X_val_mat = preprocesador.transform(parts["X_validation"])

    tamanio_fit = min(10_000, X_train_mat.shape[0])
    indices_fit = np.linspace(0, X_train_mat.shape[0] - 1, tamanio_fit, dtype=int)
    reductor = UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=0.1,
        metric="euclidean",
        random_state=config.random_state,
        n_jobs=1,
    )
    reductor.fit(X_train_mat[indices_fit])
    proyeccion = reductor.transform(X_val_mat)

    y_val = parts["y_validation"].to_numpy()
    negativos = y_val == 0
    positivos = y_val == 1

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.scatter(
        proyeccion[negativos, 0],
        proyeccion[negativos, 1],
        s=3,
        alpha=0.25,
        color="#8c8c8c",
        label="A tiempo",
        rasterized=True,
    )
    ax.scatter(
        proyeccion[positivos, 0],
        proyeccion[positivos, 1],
        s=6,
        alpha=0.55,
        color="#c0392b",
        label="Retraso",
        rasterized=True,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", frameon=False, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    fig.savefig(FIG_DIR / "umap_scatter.pdf", dpi=200)
    plt.close(fig)


def main() -> None:
    figura_mae()
    print("  mae_ic95.pdf listo")
    figura_roc_auc()
    print("  roc_auc_ic95.pdf listo")
    figura_importancia()
    print("  permutation_importance.pdf listo")
    figura_umap_scatter()
    print("  umap_scatter.pdf listo")


if __name__ == "__main__":
    main()
