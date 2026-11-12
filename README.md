# LogisTech — Predicción del tiempo de entrega en Olist

Proyecto de **Modelos y Simulación de Sistemas II** (Universidad de Antioquia).

**Autores:** Cristian Echeverry, Esteban Andrés Castaño, Mateo Aguirre.

## Entregables

| Nº | Contenido | Documento | Notebook principal |
|----|-----------|-----------|--------------------|
| I  | Formulación del problema y estado del arte | [`Entregable_I.pdf`](Entregable_I.pdf) · [`report/entregable_i.tex`](report/entregable_i.tex) | [`notebook/01_eda.ipynb`](notebook/01_eda.ipynb) |
| II | Modelos, validación y resultados | [`report/entregable_ii.tex`](report/entregable_ii.tex) | [`notebook/03_entregable_ii.ipynb`](notebook/03_entregable_ii.ipynb) (pendiente) |

Los notebooks son el artefacto reproducible que se revisa. Los scripts en
[`scripts/`](scripts/) son utilidades para correr el pipeline fuera del
cuaderno; no son objeto de evaluación.

## Pregunta de investigación

> ¿Puede predecirse, desde el momento de la compra, cuánto tardará un pedido
> y si llegará después de la fecha estimada?

Dos tareas supervisadas:

1. **Regresión** — `delivery_time_days`: días entre compra y entrega.
2. **Clasificación binaria** — `late_delivery`: 1 si la entrega real supera
   la fecha estimada, 0 en caso contrario.

## Dataset

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce):
99 441 pedidos realizados en Brasil entre 2016 y 2018. No se versiona por
tamaño; instrucciones de descarga en [`data/README.md`](data/README.md).

## Estructura del repositorio

```
.
├── Entregable_I.pdf
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── raw/                              # CSV de Olist (ignorado por Git)
├── notebook/                             # artefactos de revisión
│   ├── 01_eda.ipynb
│   ├── 02_modelado_inicial.ipynb
│   ├── 03_entregable_ii.ipynb            # (se consolida con el experimento ganador)
│   └── experimentos/
│       └── 01_baseline_pre_compra.ipynb
├── report/
│   ├── entregable_i.tex
│   ├── entregable_ii.tex
│   └── references.bib
├── src/                                  # biblioteca reutilizable
│   ├── preprocessing.py                  # carga y agregación de Olist
│   ├── encoders.py                       # FrequencyEncoder + preprocesador
│   ├── splits.py                         # muestreo temporal y 60/20/20
│   ├── models.py                         # ExperimentConfig y catálogo
│   ├── evaluation.py                     # CV, bootstrap, importancia, PCA/UMAP
│   └── experiments.py                    # run_task: orquestador
├── scripts/
│   ├── experimentos/
│   │   └── 01_baseline_pre_compra.py     # baseline con predictores pre-compra
│   └── generate_report_figures.py
├── outputs/                              # resultados locales (CSV + README)
│   ├── entregable_ii/                    # resultado FINAL (el que se califica)
│   └── experimentos/                     # intentos aislados con su README
│       └── 01_baseline_pre_compra/
└── tests/
    ├── test_pipeline.py                  # integridad de piezas
    └── test_smoke.py                     # end-to-end con SMOKE=1
```

### Carpeta `outputs/`

Los CSV y resúmenes de cada corrida viven aquí. Están ignorados por Git,
salvo los `README.md` que documentan cada experimento.

- `outputs/entregable_ii/` — resultados del modelo final, los que entran al
  reporte y la calificación.
- `outputs/experimentos/NN_<nombre>/` — intentos aislados y reproducibles
  (baseline, variantes de features, etc.), cada uno con su propio
  `README.md` que explica configuración, resultados y qué aprendimos.

## Reproducir el análisis

1. Crear entorno e instalar dependencias:

   ```bash
   python -m venv .venv
   source .venv/bin/activate              # Windows: .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

2. Descargar los CSV de Olist en `data/raw/` (ver [`data/README.md`](data/README.md)):

   ```bash
   kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip
   ```

3. Correr los notebooks en orden desde `notebook/`.

Para ejecutar el baseline sin abrir Jupyter:

```bash
python scripts/experimentos/01_baseline_pre_compra.py --max-rows 15000
```

## Pruebas

```bash
python -m unittest tests.test_pipeline   # rápidas
SMOKE=1 python -m unittest tests.test_smoke   # smoke end-to-end
```

## Compilar el reporte

Desde `report/`:

```bash
latexmk -pdf entregable_i.tex            # o entregable_ii.tex
```

También puede importarse la carpeta `report/` en Overleaf.
