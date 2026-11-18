# Experimento 02 — Distancia geográfica cliente↔vendedor

Extensión del baseline (experimento 01) con las coordenadas del cliente y
la distancia física a los vendedores del pedido, derivadas de
`olist_geolocation_dataset`. La hipótesis es que en un país tan extenso
como Brasil la distancia explica una parte estructural del tiempo de
entrega y debería mover la métrica sobre el baseline.

## Cómo reproducir

```bash
python scripts/experimentos/02_distancia_geografica.py \
    --max-rows -1 --cv-splits 3 --bootstrap 500
```

Salida: ocho CSV + `summary.json` en esta carpeta. El pipeline, la
partición cronológica y las cinco familias de modelos son los mismos del
baseline; solo cambia el conjunto de predictores. Las cifras que siguen
corresponden al dataset completo (≈96 500 pedidos útiles).

## Qué se agregó

- `customer_lat`, `customer_lon` — coordenada mediana del prefijo postal
  del cliente.
- `mean_customer_seller_distance_km` — promedio de las distancias
  ortodrómicas a los vendedores del pedido.
- `max_customer_seller_distance_km` — la mayor de esas distancias.
- `sellers_with_coordinates` — número de vendedores con coordenada válida.

Todas se calculan al momento de la compra y no introducen fuga. La
cobertura es del **98.7 %** de los pedidos (los NaN se dejan pasar como
faltantes al pipeline).

## Resultados

### Regresión (`delivery_time_days`)

| Modelo | Test MAE (días) | IC95% | Δ vs baseline |
|--------|-----------------|-------|---------------|
| **LinearSVR (campeón)** | **3.92** | [3.87, 3.98] | −0.06 |
| Random Forest | 4.14 | [4.09, 4.20] | −0.03 |
| MLP | 4.36 | [4.30, 4.42] | +0.59 |
| KNN | 4.61 | [4.56, 4.68] | +0.15 |
| **Baseline trivial (mediana)** | 4.96 | [4.91, 5.02] | — |
| Ridge | 5.06 | [5.00, 5.12] | −0.01 |

La mejora del campeón sobre el baseline del experimento 01 es de **0.06
días (−1.4 %)** y queda dentro del margen bootstrap. Lo interesante no
es tanto la magnitud como lo que ocurre con la importancia por
permutación: la distancia cliente↔vendedor se convierte en la variable
dominante del modelo.

### Clasificación (`late_delivery`)

| Modelo | Test ROC-AUC | IC95% | Δ vs baseline |
|--------|--------------|-------|---------------|
| **Random Forest (campeón por CV)** | **0.533** | [0.515, 0.550] | −0.013 |
| Logística | 0.727 | [0.714, 0.741] | +0.012 |
| LinearSVC | 0.721 | [0.708, 0.735] | +0.012 |
| MLP | 0.626 | [0.610, 0.641] | −0.049 |
| KNN | 0.571 | [0.553, 0.587] | +0.008 |
| Baseline trivial (clase mayoritaria) | 0.500 | [0.500, 0.500] | — |

El campeón por score de validación sigue por debajo de los modelos
lineales en test: Logística alcanza 0.727 —el techo observado— mientras
que Random Forest, elegido por la validación cruzada, se queda en 0.533
apenas por encima del baseline trivial. La familia lineal generaliza
mejor al periodo de prueba que la ensamble, y ese es el hilo central
del experimento 03.

### Importancia por permutación (sobre validación)

**Regresión:**

| Variable | Importancia |
|---|---|
| `mean_customer_seller_distance_km` | **2.484** |
| `max_customer_seller_distance_km` | 0.627 |
| `customer_state` | 0.536 |
| `estimated_delivery_days` | 0.486 |
| `product_category_name` | 0.100 |

La distancia promedio al vendedor se vuelve la variable **más
importante**, por encima del tiempo prometido por la plataforma y del
estado del cliente. `customer_state` pasa de 0.819 en el baseline a
0.536: la distancia absorbe buena parte de la señal geográfica que
antes se atribuía a la variable categórica.

**Clasificación:**

| Variable | Importancia |
|---|---|
| `estimated_delivery_days` | 0.094 |
| `purchase_month` | 0.063 |
| `customer_state` | 0.044 |
| `total_freight_value` | 0.024 |
| `product_category_name` | 0.022 |
| `customer_lat` | 0.020 |
| `customer_lon` | 0.020 |
| `mean_customer_seller_distance_km` | 0.018 |

Las variables geográficas entran en el top 8 pero ninguna domina. La
señal sigue repartida y la fecha prometida por Olist conserva el primer
lugar.

### Reducción de dimensión

PCA retiene el 95 % de la varianza con 24 componentes; UMAP comprime a
10. Como en el baseline, ninguna reducción mejora al pipeline completo
en test: PCA pierde 0.04 días de MAE en LinearSVR (3.96 vs 3.92) y
mantiene ROC-AUC similar en Logística (0.724 vs 0.727). UMAP a 10
dimensiones destruye la señal en ambos modelos (ROC-AUC 0.39–0.44),
consistente con un espacio donde las categóricas dominan y la métrica
euclidiana sobre embeddings densos no preserva la geometría relevante.

## Lectura de los resultados

1. **El hallazgo central no es la métrica, es la importancia.** La
   distancia cliente↔vendedor se convierte en la variable más
   informativa de la regresión. Esto refuerza el supuesto geográfico
   del problema y justifica la inclusión del enriquecimiento en los
   experimentos siguientes.

2. **La mejora cuantitativa es modesta.** −0.06 días en MAE y +0.025
   en ROC-AUC. Los IC95 se solapan con los del baseline, así que la
   diferencia no es estadísticamente clara aun con el dataset completo.

3. **El shift temporal sigue sin resolverse.** La distancia no corrige
   el problema que ya se identificó en el experimento 01: la validación
   tiene el doble de retrasos que el test, y el criterio de selección
   por score de desarrollo privilegia modelos que no generalizan al
   periodo final. Esto hay que atacarlo con una estrategia de selección
   robusta frente al shift, no con más variables.

## Rol en el proyecto

El experimento 02 queda como la **mejor versión disponible del
pipeline con señales estructurales del momento de la compra**. Su
LinearSVR es el campeón de regresión que reporta el Entregable II
(MAE 3.92 días en test). Sus salidas son también la referencia que
usa el experimento 03 para explorar la selección robusta frente al
shift temporal.
