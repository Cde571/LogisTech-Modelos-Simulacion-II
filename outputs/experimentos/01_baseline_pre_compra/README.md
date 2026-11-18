# Experimento 01 — Baseline pre-compra

Primer intento del Entregable II utilizando exclusivamente los predictores
disponibles en el **momento de la compra** sobre la agregación de pedidos de
Olist. Es el punto de partida contra el que se miden las mejoras posteriores.

## Cómo reproducir

```bash
python scripts/experimentos/01_baseline_pre_compra.py \
    --max-rows -1 --cv-splits 3 --bootstrap 500
```

Salidas: los nueve archivos de esta carpeta (ocho CSV + `summary.json`).
Para una verificación rápida de la mecánica se puede usar `--max-rows
15000`; las cifras del informe corresponden al dataset completo.

## Configuración

- Partición 60 / 20 / 20 estrictamente cronológica sobre todos los
  pedidos útiles (≈96 500).
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
- UMAP corre en modo no supervisado para evitar que la etiqueta filtre
  en el reductor.

## Resultados

### Regresión (`delivery_time_days`)

| Modelo | Test MAE (días) | IC95% MAE |
|--------|-----------------|-----------|
| **LinearSVR (campeón)** | **3.98** | [3.92, 4.04] |
| MLP | 3.77 | [3.71, 3.83] |
| Random Forest | 4.17 | [4.11, 4.22] |
| KNN | 4.47 | [4.41, 4.53] |
| **Baseline trivial (mediana)** | **4.96** | [4.91, 5.02] |
| Ridge | 5.07 | [5.01, 5.13] |

El campeón mejora al baseline trivial en **0.98 días** (−19.8 %). La
mediana se reporta como línea obligatoria: sirve para medir si cada
modelo aporta sobre la decisión más simple disponible al momento de la
compra. MLP obtiene en test un MAE menor, pero su mejor configuración
quedó por debajo de LinearSVR en la validación cruzada —el criterio de
selección por score de desarrollo— y por eso no queda como campeón.

### Clasificación (`late_delivery`)

| Modelo | Test ROC-AUC | IC95% ROC-AUC |
|--------|--------------|---------------|
| **MLP (campeón)** | **0.675** | [0.660, 0.688] |
| Logística | 0.715 | [0.701, 0.729] |
| LinearSVC | 0.709 | [0.696, 0.723] |
| KNN | 0.563 | [0.546, 0.581] |
| Random Forest | 0.546 | [0.529, 0.564] |
| **Baseline trivial (clase mayoritaria)** | **0.500** | [0.500, 0.500] |

El campeón se elige por el score sobre validación; MLP gana en ese
criterio, pero Logística y LinearSVC generalizan claramente mejor en
test. El baseline trivial se mantiene en 0.500 por construcción y
marca el piso contra el que cualquier modelo tiene que justificarse.

### Reducción de dimensión

PCA retiene el 95 % de la varianza con 24 componentes; UMAP comprime a 10.
Ninguna reducción mejora al pipeline completo en test:

| Reducción | Modelo | Test MAE / ROC-AUC |
|-----------|--------|--------------------|
| PCA_95pct | LinearSVR | MAE 4.13 |
| PCA_95pct | MLP | MAE 4.06 |
| UMAP_10d | LinearSVR | MAE 4.84 |
| UMAP_10d | MLP | MAE 5.47 |
| PCA_95pct | Random Forest | AUC 0.570 |
| UMAP_10d | Random Forest | AUC 0.396 |

PCA mantiene estructura utilizable; UMAP a 10 dimensiones destruye más
señal de la que concentra, consistente con un espacio donde las
variables categóricas dominan y la métrica euclidiana sobre embeddings
densos no preserva la geometría relevante.

### Importancia por permutación (sobre validación)

- **Regresión**: `customer_state` (0.819) y `estimated_delivery_days`
  (0.708) acaparan casi toda la señal, seguidas de `payment_value`
  (0.367), `total_price` (0.302) y `total_freight_value` (0.118). El
  estado del cliente aparece como factor estructural —consistente con
  un país grande y heterogéneo— y justifica el siguiente experimento
  con distancia geográfica.
- **Clasificación**: ninguna variable supera 0.13; `customer_state`
  (0.128), `estimated_delivery_days` (0.111) y `total_freight_value`
  (0.074) son las tres más informativas. La señal está más repartida
  que en regresión.

## Lectura de los resultados

1. **Distribution shift temporal.** La tasa de pedidos tardíos difiere
   entre particiones: train 0.079 / validation 0.116 / test 0.053. La
   validación tiene el doble de retrasos que el test, así que optimizar
   por score de validación **no coincide con la métrica real** en el
   periodo final.

2. **Selección por score de validación bajo shift.** En clasificación
   gana MLP por validación (0.721) pero Logística generaliza mejor en
   test (0.715 vs 0.675). La regularización fuerte y los pesos de
   clase de la familia lineal resisten mejor el cambio de distribución
   que la red neuronal.

3. **Set de predictores limitado al momento de la compra.** La fuerza
   de `customer_state` sugiere una señal geográfica estructural que
   motiva directamente el siguiente experimento: enriquecer con
   distancia cliente↔vendedor derivada de `olist_geolocation`.

4. **Margen moderado sobre el baseline trivial.** El campeón de
   regresión mejora casi un día al predictor por mediana (−19.8 %); el
   de clasificación añade 17 puntos de ROC-AUC al clasificador
   mayoritario. Son mejoras reales pero dejan claro que falta señal
   estructural adicional.

## Rol en el proyecto

Este experimento fija la línea base del Entregable II y entrega la
hipótesis que mueve al experimento 02: la dominancia de
`customer_state` sugiere una señal geográfica no explotada que la
siguiente iteración investiga con las coordenadas disponibles en
`olist_geolocation`.
