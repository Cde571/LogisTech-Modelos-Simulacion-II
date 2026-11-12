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
- Cinco familias de modelos exigidas por la rúbrica:
  Ridge/Logística, KNN, Random Forest, MLP, LinearSVR/LinearSVC.

## Resultados

### Regresión (`delivery_time_days`)

| Modelo | Test MAE (días) | Test RMSE | Test R² | IC95% MAE |
|--------|-----------------|-----------|---------|-----------|
| **LinearSVR (campeón)** | **4.69** | 6.18 | **−0.065** | [4.54, 4.83] |
| KNN | 5.11 | 6.49 | −0.175 | [4.96, 5.26] |
| MLP | 5.59 | 7.21 | −0.450 | [5.42, 5.75] |
| Ridge | 6.40 | 7.69 | −0.648 | [6.23, 6.55] |
| Random Forest | 4.76 | 6.32 | −0.114 | [4.61, 4.91] |

### Clasificación (`late_delivery`)

| Modelo | Test ROC-AUC | Test F1 | Recall | IC95% ROC-AUC |
|--------|--------------|---------|--------|---------------|
| **Random Forest (campeón)** | **0.487** | 0.071 | 0.108 | [0.446, 0.528] |
| KNN | 0.488 | 0.000 | 0.000 | [0.449, 0.527] |
| Logística | 0.720 | 0.130 | 0.970 | [0.689, 0.752] |
| LinearSVC | 0.718 | 0.133 | 0.970 | [0.687, 0.750] |
| MLP | 0.405 | 0.000 | 0.000 | [0.352, 0.452] |

El modelo campeón se eligió por la **media del score de validación cruzada**.
En clasificación ese criterio premió a Random Forest, pero sobre el test la
Logística generaliza mucho mejor (0.720 vs 0.487) — un síntoma claro de
*distribution shift*.

### Reducción de dimensión

PCA retiene 94 % de la varianza con 24 componentes; UMAP comprime a 10.
Ninguna reducción mejora al pipeline completo en test, lo cual indica que
el cuello de botella no es la dimensionalidad.

### Importancia por permutación

- **Regresión**: `estimated_delivery_days` (0.695) + `customer_state` (0.261)
  acaparan casi toda la señal. El resto de predictores aporta < 3 %.
- **Clasificación**: ninguna variable supera 0.09; `estimated_delivery_days`
  es la única con aporte positivo. Varios features arrojan importancia
  negativa (hacen daño por ruido).

## Diagnóstico — por qué no dio lo esperado

1. **Distribution shift temporal.** La tasa de pedidos tardíos difiere entre
   particiones: train 0.072 / validation 0.122 / test 0.056. La validación
   tiene el doble de retrasos que el test, así que optimizar por score de
   validación **no coincide con la métrica real** en el periodo final.

2. **Selección por media del CV en condiciones de shift.** Random Forest
   quedó campeón por tener la mejor media de validación, pero sobre test
   colapsa a azar (AUC 0.487). La Logística, con regularización fuerte y
   pesos de clase, mantiene AUC 0.720 — era el modelo correcto pero la
   validación lo descartó.

3. **Set de predictores limitado.** Solo variables pre-compra. Faltan
   señales con fuerte asociación teórica a demora:
   - **Distancia geográfica** cliente↔vendedor (sin `olist_geolocation`).
   - **Historial del vendedor**: tiempo de preparación previo, tasa de
     retraso acumulada antes de la compra (expanding window).
   - **Carga del sistema**: pedidos simultáneos activos en el momento.

4. **Objetivo R² negativo en test.** El tiempo de entrega varía poco pedido
   a pedido pero sí entre periodos (festivos, eventos). Sin variables que
   capturen ese contexto, el modelo no supera a la media del test.

## Rol en el proyecto

Este experimento **no es el resultado final** del Entregable II. Se conserva
para documentar la línea base y justificar las extensiones del experimento
siguiente. Los resultados definitivos del entregable viven en
[`outputs/entregable_ii/`](../../entregable_ii/).
