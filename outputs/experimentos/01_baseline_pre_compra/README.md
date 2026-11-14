# Experimento 01 — Baseline pre-compra

Primer intento del Entregable II utilizando exclusivamente los predictores
disponibles en el **momento de la compra** sobre la agregación de pedidos de
Olist. Es el punto de partida contra el que se miden las mejoras posteriores.

## Cómo reproducir

```bash
python scripts/experimentos/01_baseline_pre_compra.py \
    --max-rows 15000 --cv-splits 3 --bootstrap 500
```

Salidas: los nueve archivos de esta carpeta (ocho CSV + `summary.json`). Para
ejecutar sobre todo el dataset usar `--max-rows -1`.

## Configuración

- Muestra temporal: 15 000 pedidos (submuestreo cronológico uniforme).
- Partición 60 / 20 / 20 estrictamente cronológica.
- Validación cruzada: `TimeSeriesSplit` con 3 particiones (solo sobre train).
- Bootstrap de 500 iteraciones para IC95% sobre el test.
- Codificación diferenciada por cardinalidad (umbrales 30 / 100):
  - Baja → one-hot plano.
  - Media → one-hot con `min_frequency=20`.
  - Alta (`customer_city`, `product_category_name`) → `FrequencyEncoder`.
- Catálogo evaluado: **baseline trivial** (mediana / clase mayoritaria) más
  las cinco familias exigidas por la rúbrica: Ridge/Logística, KNN, Random
  Forest, MLP, LinearSVR/LinearSVC.
- Importancia por permutación medida sobre **validación**; test queda
  reservado únicamente para la evaluación final.
- UMAP corre en modo no supervisado (la revisión anterior había filtrado `y`
  al reductor, lo que explicaba ROC-AUC ~1.0 en entrenamiento).

## Resultados

### Regresión (`delivery_time_days`)

| Modelo | Test MAE (días) | IC95% MAE |
|--------|-----------------|-----------|
| **LinearSVR (campeón)** | **4.69** | [4.54, 4.83] |
| Random Forest | 4.76 | [4.61, 4.91] |
| **Baseline trivial (mediana)** | **5.04** | [4.91, 5.18] |
| KNN | 5.11 | [4.96, 5.26] |
| MLP | 5.59 | [5.42, 5.75] |
| Ridge | 6.40 | [6.23, 6.55] |

El campeón mejora al baseline trivial en **0.35 días** (−6.9 %). La magnitud
es modesta y explica por qué el asesor pidió reportar la mediana como
referencia: sin esa línea, el error absoluto del modelo parece aceptable en
términos operativos pero se queda cerca de simplemente predecir la mediana.

### Clasificación (`late_delivery`)

| Modelo | Test ROC-AUC | IC95% ROC-AUC |
|--------|--------------|---------------|
| **Random Forest (campeón por CV)** | **0.487** | [0.446, 0.528] |
| KNN | 0.488 | [0.449, 0.527] |
| **Baseline trivial (clase mayoritaria)** | **0.500** | [0.500, 0.500] |
| Logística | 0.720 | [0.689, 0.752] |
| LinearSVC | 0.718 | [0.687, 0.750] |
| MLP | 0.405 | [0.352, 0.452] |

El modelo campeón se eligió por la **media del score de validación cruzada**.
Ese criterio premió a Random Forest, pero sobre test no supera al baseline
trivial (0.487 vs 0.500). La Logística, en cambio, generaliza bien (0.720)
— pero el proceso de selección la descartó.

### Reducción de dimensión

PCA retiene el 95 % de la varianza con 24 componentes; UMAP comprime a 10.
Ninguna reducción mejora al pipeline completo en test:

| Reducción | Modelo | Test MAE / ROC-AUC |
|-----------|--------|--------------------|
| PCA_95pct | LinearSVR | MAE 4.91 |
| UMAP_10d | LinearSVR | MAE 4.93 |
| PCA_95pct | Random Forest | AUC 0.603 |
| UMAP_10d | Random Forest | AUC 0.356 |

PCA mantiene estructura utilizable; UMAP a 10 dimensiones destruye más señal
de la que concentra, consistente con un espacio donde las variables
categóricas dominan y la métrica euclidiana sobre embeddings densos no
preserva la geometría relevante.

### Importancia por permutación (sobre validación)

- **Regresión**: `customer_state` (0.744) y `estimated_delivery_days` (0.735)
  acaparan casi toda la señal, seguidas de lejos por `product_category_name`
  (0.088) y `total_freight_value` (0.080). El estado del cliente aparece
  como factor estructural —consistente con un país grande y heterogéneo— y
  justifica el siguiente experimento con distancia geográfica.
- **Clasificación**: ninguna variable supera 0.13; `customer_state` (0.127),
  `estimated_delivery_days` (0.083) y `total_freight_value` (0.077) son las
  tres más informativas. La señal está más repartida que en regresión.

## Diagnóstico — por qué no dio lo esperado

1. **Distribution shift temporal.** La tasa de pedidos tardíos difiere entre
   particiones: train 0.072 / validation 0.122 / test 0.056. La validación
   tiene el doble de retrasos que el test, así que optimizar por score de
   validación **no coincide con la métrica real** en el periodo final.

2. **Selección por media del CV en condiciones de shift.** Random Forest
   quedó campeón por tener la mejor media de validación, pero sobre test
   colapsa a azar (AUC 0.487, peor que el baseline trivial 0.500). La
   Logística, con regularización fuerte y pesos de clase, mantiene AUC
   0.720 — era el modelo correcto pero la validación lo descartó.

3. **Set de predictores limitado.** Solo variables pre-compra. Faltan
   señales con fuerte asociación teórica a demora:
   - **Distancia geográfica** cliente↔vendedor (sin `olist_geolocation`).
     Es el siguiente experimento, motivado por el peso de `customer_state`.
   - **Historial del vendedor**: tiempo de preparación previo, tasa de
     retraso acumulada antes de la compra (expanding window).
   - **Carga del sistema**: pedidos simultáneos activos en el momento.

4. **La mejora sobre el baseline trivial es pequeña.** El campeón de
   regresión mejora 0.35 días al predictor por mediana; el campeón de
   clasificación no mejora al predictor mayoritario. Esto marca el piso
   contra el que cualquier extensión tiene que justificarse.

## Rol en el proyecto

Este experimento **no es el resultado final** del Entregable II. Documenta
la línea base y motiva las extensiones posteriores (experimentos 02 en
adelante) con hallazgos concretos: shift temporal, dominancia de
`customer_state`, bajo margen sobre el baseline trivial.
