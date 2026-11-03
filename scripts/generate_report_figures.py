from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "entregable_ii"
FIGURES = ROOT / "report" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)

reg = pd.read_csv(RESULTS / "regression_ranking.csv")
cls = pd.read_csv(RESULTS / "classification_ranking.csv")

fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
sns.barplot(data=reg, x="test_MAE", y="model", color="#4472C4", ax=axes[0])
axes[0].set(title="Regresion en prueba", xlabel="MAE [dias]", ylabel="")
sns.barplot(data=cls, x="test_ROC_AUC", y="model", color="#70AD47", ax=axes[1])
axes[1].set(title="Clasificacion en prueba", xlabel="ROC-AUC", ylabel="")
axes[1].set_xlim(0, 0.8)
fig.tight_layout()
fig.savefig(FIGURES / "model_results.png", dpi=300, bbox_inches="tight")
plt.close(fig)

reg_red = pd.read_csv(RESULTS / "regression_reductions.csv")
cls_red = pd.read_csv(RESULTS / "classification_reductions.csv")
reg_red["configuration"] = reg_red["reduction"] + " + " + reg_red["model"]
cls_red["configuration"] = cls_red["reduction"] + " + " + cls_red["model"]

fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8))
sns.barplot(data=reg_red, x="MAE", y="configuration", hue="reduction", legend=False, ax=axes[0])
axes[0].axvline(reg.iloc[0]["test_MAE"], color="black", linestyle="--", linewidth=1)
axes[0].set(title="Reduccion: regresion", xlabel="MAE [dias]", ylabel="")
sns.barplot(data=cls_red, x="ROC_AUC", y="configuration", hue="reduction", legend=False, ax=axes[1])
axes[1].axvline(cls["test_ROC_AUC"].max(), color="black", linestyle="--", linewidth=1)
axes[1].set(title="Reduccion: clasificacion", xlabel="ROC-AUC", ylabel="")
axes[1].set_xlim(0, 0.8)
fig.tight_layout()
fig.savefig(FIGURES / "reduction_results.png", dpi=300, bbox_inches="tight")
plt.close(fig)
