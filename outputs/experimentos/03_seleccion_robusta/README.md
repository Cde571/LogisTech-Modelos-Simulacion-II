# Experimento 03 — Selección de modelos robusta al shift temporal

Mismo pipeline enriquecido del experimento 02 (baseline + distancia
geográfica), pero cambia el **criterio de selección**: en vez de la media
del score entre los tres pliegues de la validación cruzada temporal
usamos el *peor* pliegue. La intuición, prestada de la optimización
robusta y del uso de CV para datos con dependencia temporal (Bergmeir &
Benítez, 2012), es que un modelo cuya peor corrida es buena generaliza
mejor cuando la distribución se mueve en el tiempo.

## Cómo reproducir

```bash
python scripts/experimentos/03_seleccion_robusta.py \
    --max-rows 15000 --cv-splits 3 --bootstrap 500
```

Salida: ocho CSV + `summary.json` en esta carpeta. Lo único nuevo frente
al experimento 02 es `selection_criterion='worst'` en la configuración.

## Resultados

### Regresión (`delivery_time_days`)

| Modelo | CV mean (−MAE) | CV worst (−MAE) | Test MAE | IC95% |
|--------|----------------|------------------|----------|-------|
| **LinearSVR (campeón)** | −5.59 | **−6.12** | 4.61 | [4.47, 4.74] |
| KNN | −5.96 | −6.62 | 5.18 | [5.02, 5.33] |
| Ridge | −5.92 | −6.70 | 6.35 | [6.18, 6.50] |
| Random Forest | −6.15 | −6.85 | 4.86 | [4.70, 5.02] |
| Baseline trivial (mediana) | −6.92 | −7.88 | 5.04 | [4.91, 5.18] |
| MLP | −7.99 | **−12.35** | 5.54 | [5.37, 5.70] |

LinearSVR vuelve a ganar. Lo interesante no es quién gana, sino cómo se
reparte la distancia entre `cv_mean` y `cv_worst`: MLP tiene un peor
pliegue un 55 % por debajo de su media, señal de que su inestabilidad
entre pliegues era alta pero el criterio por media la ocultaba. El
criterio robusto convierte esa inestabilidad en un castigo explícito.

### Clasificación (`late_delivery`)

| Modelo | CV mean (ROC-AUC) | CV worst (ROC-AUC) | Test ROC-AUC | IC95% |
|--------|-------------------|---------------------|--------------|-------|
| **Random Forest (campeón)** | 0.632 | **0.563** | 0.512 | [0.471, 0.552] |
| Logística | 0.599 | 0.557 | **0.734** | [0.704, 0.765] |
| LinearSVC | 0.590 | 0.544 | 0.729 | [0.696, 0.760] |
| KNN | 0.581 | 0.537 | 0.514 | [0.474, 0.554] |
| MLP | 0.551 | 0.510 | 0.389 | [0.342, 0.436] |
| Baseline trivial (clase mayoritaria) | 0.500 | 0.500 | 0.500 | [0.500, 0.500] |

Random Forest sigue campeón. La diferencia frente a Logística es apenas
0.006 ROC-AUC en el peor pliegue (0.563 vs 0.557), y Random Forest gana
también por media. **El criterio robusto no cambia el ranking.**

## El hallazgo que sí importa

El resultado anterior parece decepcionante pero es la pieza más honesta
del proyecto: aprendimos **dónde no está el problema**.

- Dentro del rango de desarrollo (entrenamiento + validación) la
  distribución es relativamente homogénea entre los tres pliegues de
  CV temporal. El peor pliegue cuesta algunas centésimas de ROC-AUC, no
  la caída catastrófica que sí se observa en test.
- Logística pasa de 0.629 en validación holdout a 0.734 en test. Random
  Forest hace el movimiento inverso: 0.715 → 0.512. El sobreajuste de
  Random Forest a la geografía y a la temporada se manifiesta recién en
  el test, no en los pliegues anteriores.
- Un criterio robusto a la variación *dentro* de CV no puede corregir
  un shift que solo existe *entre* desarrollo y prueba. La estructura
  del problema queda fuera del alcance de la validación cruzada.

Para que un criterio robusto moviera el campeón habría que poner el
holdout de validación *más cerca* del test en el tiempo —o aplicar una
corrección de shift (reponderación por importancia, adaptación de
dominio)—, y eso excede el alcance declarado del Entregable II.

## Conclusión

El experimento 03 cumple su rol metodológico: documenta que la elección
del criterio de selección (media vs. peor pliegue) no modifica los
campeones en este dataset. El techo con señales previas a la compra
queda establecido:

- **Regresión:** LinearSVR, MAE 4.61 días.
- **Clasificación:** Random Forest por CV, con la nota explícita de
  que Logística es el modelo que efectivamente generaliza a la ventana
  de test (ROC-AUC 0.734). La discrepancia entre ambos es el shift
  temporal, no el criterio de selección.

## Referencias

- Bergmeir, C., & Benítez, J. M. (2012). *On the use of cross-validation
  for time series predictor evaluation.* Information Sciences, 191,
  192–213. — Justifica el uso de CV temporal y discute la sensibilidad
  del criterio de agregación bajo dependencia serial.
- Varma, S., & Simon, R. (2006). *Bias in error estimation when using
  cross-validation for model selection.* BMC Bioinformatics, 7(1), 91.
  — Advierte sobre interpretar la media de CV como garantía de
  generalización; el peor pliegue es una cota complementaria.
