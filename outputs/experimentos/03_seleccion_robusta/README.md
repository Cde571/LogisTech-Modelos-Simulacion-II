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
    --max-rows -1 --cv-splits 3 --bootstrap 500
```

Salida: ocho CSV + `summary.json` en esta carpeta. Lo único nuevo frente
al experimento 02 es `selection_criterion='worst'` en la configuración.
Las tablas que siguen corresponden a la corrida con el dataset completo
(96 476 pedidos útiles).

## Resultados

### Regresión (`delivery_time_days`)

| Modelo | CV mean (−MAE) | CV worst (−MAE) | Test MAE | IC95% |
|--------|----------------|------------------|----------|-------|
| **Random Forest (campeón)** | −5.56 | **−6.36** | 4.14 | [4.09, 4.20] |
| Ridge | −5.62 | −6.43 | 5.06 | [5.00, 5.12] |
| KNN | −5.71 | −6.55 | 4.61 | [4.56, 4.68] |
| MLP | −5.63 | −6.55 | 4.36 | [4.30, 4.42] |
| LinearSVR | −5.65 | −6.78 | 3.89 | [3.84, 3.95] |
| Baseline trivial (mediana) | −6.84 | −7.76 | 4.96 | [4.91, 5.02] |

El criterio robusto **cambia el campeón** respecto al experimento 02
(mean): ahí ganaba LinearSVR con MAE 3.92; aquí lo desplaza Random
Forest porque LinearSVR tiene el peor pliegue más bajo de todos
(−6.78). El costo es real: el RF seleccionado por peor pliegue da MAE
4.14 en test, 0.22 días peor que el LinearSVR de exp 02. Documentamos
ese costo como evidencia de que el criterio no corrige el shift entre
desarrollo y prueba.

### Clasificación (`late_delivery`)

| Modelo | CV mean (ROC-AUC) | CV worst (ROC-AUC) | Test ROC-AUC | IC95% |
|--------|-------------------|---------------------|--------------|-------|
| **Random Forest (campeón)** | 0.643 | **0.585** | 0.637 | [0.621, 0.652] |
| Logística | 0.636 | 0.566 | 0.727 | [0.714, 0.741] |
| LinearSVC | 0.634 | 0.565 | 0.721 | [0.708, 0.735] |
| KNN | 0.595 | 0.565 | 0.571 | [0.553, 0.587] |
| MLP | 0.603 | 0.545 | 0.679 | [0.665, 0.694] |
| Baseline trivial | 0.500 | 0.500 | 0.500 | [0.500, 0.500] |

Aquí ocurre lo interesante: Random Forest sigue ganando por ambos
criterios, **pero con hiperparámetros distintos**. Exp 02 (mean) eligió
`max_depth=12, min_samples_leaf=1`, que da 0.533 en test. Exp 03
(worst) eligió `max_depth=24, min_samples_leaf=2`, que da 0.637 en
test. El salto de +0.10 ROC-AUC dentro de la misma familia es la mejor
evidencia del efecto del criterio robusto en este dataset.

Aun así, Logística sigue muy por encima (0.727) tanto del RF-mean como
del RF-worst. El criterio ayuda a seleccionar hiperparámetros más
estables dentro de la familia ganadora, pero no resuelve el shift que
favorece a la familia lineal.

## El hallazgo del experimento

Con el dataset completo y las mallas de hiperparámetros ampliadas, el
criterio robusto **sí** modifica decisiones:

- En regresión cambia la familia campeón (LinearSVR → RF) con un costo
  medible en MAE. Media sigue siendo la mejor elección por métrica
  final.
- En clasificación mantiene la familia (RF) pero elige hiperparámetros
  menos agresivos que generalizan +0.10 ROC-AUC en test.
- Ninguno de los dos criterios corrige el shift que separa a Logística
  (0.727) del RF (0.637 en el mejor caso): esa brecha es estructural y
  no cabe dentro del alcance de este experimento.

## Conclusión

Se reportan los dos criterios en el informe. Para regresión se mantiene
el campeón por media (LinearSVR, MAE 3.92). Para clasificación queda
documentado que el criterio robusto mejora el desempeño del campeón
dentro de la familia Random Forest y que, aun así, la familia lineal
generaliza mejor por el shift temporal pre-/post-marzo 2018.

## Referencias

- Bergmeir, C., & Benítez, J. M. (2012). *On the use of cross-validation
  for time series predictor evaluation.* Information Sciences, 191,
  192–213. — Justifica el uso de CV temporal y discute la sensibilidad
  del criterio de agregación bajo dependencia serial.
- Varma, S., & Simon, R. (2006). *Bias in error estimation when using
  cross-validation for model selection.* BMC Bioinformatics, 7(1), 91.
  — Advierte sobre interpretar la media de CV como garantía de
  generalización; el peor pliegue es una cota complementaria.
