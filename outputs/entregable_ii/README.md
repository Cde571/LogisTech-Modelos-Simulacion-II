# Salidas del Entregable II

Carpeta donde el cuaderno `notebook/03_entregable_ii.ipynb` consolida las
piezas numéricas del entregable. No se versionan los CSV (el `.gitignore`
del repo solo conserva los README): se regeneran ejecutando ese cuaderno
después de correr los tres experimentos.

## Cómo reproducir

```bash
# Primero los tres experimentos (dataset completo: --max-rows -1)
python scripts/experimentos/01_baseline_pre_compra.py --max-rows -1
python scripts/experimentos/02_distancia_geografica.py --max-rows -1
python scripts/experimentos/03_seleccion_robusta.py   --max-rows -1

# Luego el cuaderno de síntesis
jupyter nbconvert --to notebook --execute notebook/03_entregable_ii.ipynb \
    --output 03_entregable_ii.ipynb
```

Con el dataset completo (≈96 500 pedidos útiles) cada experimento tarda
entre 10 y 25 minutos en una laptop común. Si solo se quiere una
validación rápida de la mecánica, basta con `--max-rows 15000`; el
informe reporta las cifras del dataset completo.

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
- `shift_por_particion.json` — número de pedidos, rango de fechas y
  tasa de retraso observada en train / validación / test. Documenta el
  shift temporal entre particiones.
- `summary.json` — resumen ejecutivo con el campeón elegido por
  experimento, los intervalos de confianza y la lista de candidatas a
  eliminar por bajo aporte.

## Decisiones reflejadas en el resumen

- **Campeón de regresión:** LinearSVR del experimento 02, MAE **3.92
  días** en test (IC95 [3.87, 3.98]). La distancia cliente-vendedor
  pasa a ser la variable más informativa.
- **Campeón de clasificación por CV:** Random Forest del experimento
  02, ROC-AUC **0.533** en test. La nota obligatoria es que Logística
  —no elegida por el criterio— alcanza **0.727** en test porque
  generaliza mejor al shift temporal; esa discrepancia se discute en el
  informe como limitación conocida.
- **Lo que aporta el experimento 03:** con el dataset completo y mallas
  ampliadas, el criterio por peor pliegue sí modifica decisiones:
  mantiene la familia RF para clasificación pero elige hiperparámetros
  más estables (ROC-AUC test 0.637, +0.10 frente a la media); en
  regresión desplaza a LinearSVR por RF con un costo de 0.22 días de
  MAE. Ninguno de los dos criterios cierra la brecha de shift que
  separa a Logística del RF.
