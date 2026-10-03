# LogisTech — Entregable I

Proyecto de **Modelos y Simulación de Sistemas II** (Universidad de Antioquia).

Este repositorio contiene el **Entregable I**: formulación del problema, descripción de la base de datos y estado del arte para un sistema de predicción del tiempo de entrega en pedidos de comercio electrónico.

**Autores:** Cristian Echeverry, Esteban Andrés Castaño, Mateo Aguirre.

## Documento del proyecto

**→ [Entregable_I.pdf](Entregable_I.pdf)**

Fuente LaTeX editable: [`report/entregable_i.tex`](report/entregable_i.tex). Referencias bibliográficas: [`report/references.bib`](report/references.bib).

## Pregunta de investigación

> ¿Puede predecirse, desde el momento de la compra, cuánto tardará un pedido y si llegará después de la fecha estimada?

Se formulan dos tareas supervisadas complementarias:

1. **Regresión:** predecir `delivery_time_days`, el tiempo entre la compra y la entrega.
2. **Clasificación binaria:** predecir `late_delivery`, que vale 1 cuando la entrega real ocurre después de la fecha estimada y 0 en caso contrario.

## Dataset

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): 99 441 pedidos realizados en Brasil entre 2016 y 2018, distribuidos en nueve archivos CSV relacionados por identificadores.

El dataset no se versiona por tamaño. Instrucciones de descarga en [`data/README.md`](data/README.md).

## Estructura

```
.
├── Entregable_I.pdf                     # Reporte en PDF (lectura directa)
├── README.md
├── requirements.txt
├── data/
│   ├── README.md                        # Instrucciones de descarga de Olist
│   └── raw/                             # CSV de Olist (ignorado por Git)
├── notebook/
│   └── 01_eda.ipynb                     # Análisis exploratorio reproducible
├── report/
│   ├── entregable_i.tex                 # Fuente LaTeX del reporte
│   └── references.bib                   # Bibliografía
└── src/
    └── preprocessing.py                 # Carga, agregación y targets de Olist
```

## Reproducir el análisis

1. Crear entorno e instalar dependencias:

   ```bash
   python -m venv .venv
   source .venv/bin/activate            # Windows: .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

2. Descargar los CSV de Olist dentro de `data/raw/` (ver [`data/README.md`](data/README.md)):

   ```bash
   kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip
   ```

3. Abrir el notebook:

   ```bash
   jupyter lab notebook/01_eda.ipynb
   ```

   Ejecuta en orden y produce la composición de las tablas, la construcción de los objetivos, el análisis de faltantes, las distribuciones y la auditoría final de predictores admisibles.

## Compilar el reporte

Desde `report/`:

```bash
latexmk -pdf entregable_i.tex
```

También puede importarse la carpeta `report/` en Overleaf.
