# Experimento 02 — Distancia geográfica cliente↔vendedor

Extensión del baseline (experimento 01) con las coordenadas del cliente y
la distancia física a los vendedores del pedido, derivadas de
`olist_geolocation_dataset`. La hipótesis es que en un país tan extenso
como Brasil la distancia explica una parte estructural del tiempo de
entrega y debería mover la métrica sobre el baseline.

## Cómo reproducir

```bash
python scripts/experimentos/02_distancia_geografica.py \
    --max-rows 15000 --cv-splits 3 --bootstrap 500
```

Salida: ocho CSV + `summary.json` en esta carpeta. El pipeline, la
partición cronológica y las cinco familias de modelos son los mismos del
baseline; solo cambia el conjunto de predictores.

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
| **LinearSVR (campeón)** | **4.61** | [4.47, 4.74] | −0.08 |
| Random Forest | 4.86 | [4.70, 5.02] | +0.10 |
| **Baseline trivial (mediana)** | 5.04 | [4.91, 5.18] | — |
| KNN | 5.18 | [5.02, 5.33] | +0.07 |
| MLP | 5.54 | [5.37, 5.70] | −0.05 |
| Ridge | 6.34 | [6.18, 6.48] | −0.06 |

La mejora del campeón sobre el baseline del experimento 01 es de **0.08
días (−1.7 %)** y queda dentro del margen bootstrap. Lo interesante no es
tanto la magnitud como lo que ocurre con la importancia por permutación.

### Clasificación (`late_delivery`)

| Modelo | Test ROC-AUC | IC95% | Δ vs baseline |
|--------|--------------|-------|---------------|
| **Random Forest (campeón por CV)** | **0.512** | [0.471, 0.552] | +0.025 |
| KNN | 0.514 | [0.474, 0.554] | +0.027 |
| Baseline trivial (clase mayoritaria) | 0.500 | [0.500, 0.500] | — |
| Logística | 0.734 | [0.704, 0.765] | +0.014 |
| LinearSVC | 0.729 | [0.696, 0.760] | +0.011 |
| MLP | 0.389 | [0.342, 0.436] | −0.016 |

El campeón por CV sigue perdiendo contra el baseline trivial en el
intervalo de confianza. La logística —que el criterio de selección
descarta— se mantiene como el modelo que mejor generaliza, con un leve
avance sobre la corrida sin distancia (0.734 vs 0.720).

### Importancia por permutación (sobre validación)

**Regresión:**

| Variable | Importancia |
|---|---|
| `mean_customer_seller_distance_km` | **1.682** |
| `estimated_delivery_days` | 0.477 |
| `customer_state` | 0.350 |
| `product_category_name` | 0.105 |
| `max_customer_seller_distance_km` | 0.100 |

La distancia promedio al vendedor se vuelve la variable **más
importante**, por encima del tiempo prometido por la plataforma. El peso
de `customer_state` cae de 0.744 (baseline) a 0.350: la distancia absorbe
buena parte de lo que antes se atribuía al estado.

**Clasificación:**

| Variable | Importancia |
|---|---|
| `estimated_delivery_days` | 0.081 |
| `purchase_month` | 0.043 |
| `customer_state` | 0.032 |
| `total_freight_value` | 0.028 |
| `customer_lon` | 0.022 |
| `max_customer_seller_distance_km` | 0.020 |
| `mean_customer_seller_distance_km` | 0.019 |

Las variables geográficas entran en el top 7 pero ninguna domina. La
señal sigue repartida y la variable prometida por Olist conserva el
primer lugar.

### Reducción de dimensión

PCA retiene el 95 % de la varianza con 25 componentes; UMAP comprime a 10.
Como en el baseline, ninguna reducción mejora al pipeline completo en
test. UMAP (no supervisado) se queda muy por debajo, consistente con un
espacio donde las categóricas dominan y la métrica euclidiana sobre
embeddings densos no preserva la geometría relevante.

## Diagnóstico

1. **El hallazgo central no es la métrica, es la importancia.** La
   distancia cliente-vendedor se convierte en la variable más informativa
   para la regresión. Esto refuerza el supuesto geográfico del problema
   y justifica la inclusión del enriquecimiento en los experimentos
   siguientes.

2. **La mejora cuantitativa es modesta.** −0.08 días en MAE y +0.025 en
   ROC-AUC. Los IC95 se solapan con los del baseline, así que la
   diferencia no es estadísticamente clara con 15 000 pedidos.

3. **El shift temporal sigue sin resolverse.** La distancia no corrige el
   problema que ya se identificó en el experimento 01: la validación
   tiene el doble de retrasos que el test, y el criterio de selección por
   score medio de CV privilegia modelos que no generalizan. Esto hay que
   atacarlo con selección por estabilidad temporal o con pesos por época,
   no con más variables.

## Rol en el proyecto

El experimento 02 queda como la **mejor versión disponible del pipeline
con señales estructurales del momento de la compra**. Sus salidas son la
referencia para comparar los experimentos 03 en adelante (selección
robusta frente a shift, histórico del vendedor, carga del sistema).
