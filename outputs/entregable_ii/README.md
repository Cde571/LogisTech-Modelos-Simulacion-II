# Salidas del Entregable II

Carpeta donde el cuaderno `notebook/03_entregable_ii.ipynb` consolida las
piezas numéricas del entregable. No se versionan los CSV (el `.gitignore`
del repo solo conserva los README): se regeneran ejecutando ese cuaderno
después de correr los tres experimentos.

## Cómo reproducir

```bash
# Primero los tres experimentos (si no se han corrido antes)
python scripts/experimentos/01_baseline_pre_compra.py --max-rows 15000
python scripts/experimentos/02_distancia_geografica.py --max-rows 15000
python scripts/experimentos/03_seleccion_robusta.py   --max-rows 15000

# Luego el cuaderno de síntesis
jupyter nbconvert --to notebook --execute notebook/03_entregable_ii.ipynb \
    --output 03_entregable_ii.ipynb
```

## Archivos que produce

- `variable_analysis_regression.csv` — correlación (Spearman / V de
  Cramér) e información mutua de cada predictor contra
  `delivery_time_days`. Marca como `drop_candidate` a las variables en
  el cuartil inferior de información mutua (II.2.1 del informe).
- `variable_analysis_classification.csv` — mismo análisis contra
  `late_delivery`.
- `experiments_comparison.csv` — campeón y métrica de prueba de los
  tres experimentos (baseline, +distancia, criterio robusto) para
  regresión y clasificación. Fuente de la tabla principal de II.1.
- `reductions_comparison.csv` — resultados sobre prueba del pipeline
  con PCA al 95 % de varianza y UMAP a 10 dimensiones, sobre los dos
  mejores modelos de cada tarea. Alimenta la discusión de II.2.
- `summary.json` — resumen ejecutivo con el campeón elegido por
  experimento, los intervalos de confianza y la lista de candidatas a
  eliminar por bajo aporte.

## Decisiones reflejadas en el resumen

- **Campeón de regresión:** LinearSVR del experimento 02, MAE 4.61
  días en test (IC95 [4.47, 4.74]). La distancia cliente-vendedor pasa
  a ser la variable más informativa.
- **Campeón de clasificación por CV:** Random Forest del experimento
  02, ROC-AUC 0.512 en test. La nota obligatoria es que Logística —no
  elegida por el criterio— alcanza 0.734 en test porque generaliza
  mejor al shift temporal; esa discrepancia se discute en el informe
  como limitación conocida y queda sustentada por el experimento 03,
  que muestra que la elección media/peor pliegue no modifica el
  ranking en este dataset.
