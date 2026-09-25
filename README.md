# LogisTech: predicción de entregas de comercio electrónico

Proyecto de **Modelos y Simulación de Sistemas II**. El repositorio deja preparado el Entregable I en formato IEEE Transactions y un flujo reproducible para estudiar la pregunta:

## Documento del proyecto

**[Abrir el informe compilado en PDF](report/main.pdf)**

El código fuente editable está en [`report/main.tex`](report/main.tex).

> ¿Es posible predecir, desde el momento de la compra, cuánto tardará un pedido y si llegará después de la fecha estimada?

## Objetivo

Se formulan dos tareas supervisadas complementarias:

1. **Regresión:** predecir `delivery_time_days`, el tiempo transcurrido entre la compra y la entrega al cliente.
2. **Clasificación binaria:** predecir `late_delivery`, que vale 1 cuando la entrega real ocurre después de la fecha estimada y 0 en caso contrario.

La unidad de análisis es el pedido. Solo se usan predictores conocidos en el momento de la compra. Las fechas de entrega, despacho y otras señales posteriores se reservan exclusivamente para construir los objetivos y se excluyen del modelado para evitar *data leakage*.

## Dataset

Se utiliza [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), que contiene cerca de 100 mil pedidos realizados entre 2016 y 2018 y varias tablas relacionadas por identificadores de pedido, cliente, producto y vendedor.

El dataset no se versiona por su tamaño y por buenas prácticas de reproducibilidad. Descárguelo con la API de Kaggle:

```bash
python -m pip install kaggle
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip
```

También puede descargarlo manualmente desde Kaggle y extraer todos los CSV dentro de `data/raw/`. Consulte [data/README.md](data/README.md) para la lista de archivos esperados.

## Estructura

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── raw/                 # CSV originales, ignorados por Git
│   └── processed/           # salidas locales, ignoradas por Git
├── notebooks/
│   ├── 01_eda_preparacion_datos.ipynb
│   ├── 02_modelado_inicial.ipynb
│   └── 03_entregable_ii_completo_colab.ipynb
├── report/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
└── src/
    └── preprocessing.py
```

## Instalación

Se recomienda Python 3.11 o posterior:

```bash
python -m venv .venv
```

Active el entorno (`.venv\Scripts\activate` en Windows o `source .venv/bin/activate` en Linux/macOS) e instale las dependencias:

```bash
python -m pip install -r requirements.txt
```

## Reproducción

1. Descargue los CSV en `data/raw/`.
2. Inicie Jupyter con `jupyter lab` desde la raíz del repositorio.
3. Ejecute en orden `notebooks/01_eda_preparacion_datos.ipynb` y `notebooks/02_modelado_inicial.ipynb`.
4. Revise las estadísticas y métricas generadas localmente; el repositorio no afirma resultados que no hayan sido ejecutados.
5. Compile el reporte desde `report/` con `latexmk -pdf main.tex` o importe la carpeta `report/` en Overleaf.

El primer notebook valida archivos, convierte fechas, construye los objetivos, agrega las tablas a nivel de pedido y analiza faltantes y distribuciones. El segundo crea particiones cronológicas y líneas base reproducibles con preprocesamiento aprendido únicamente sobre entrenamiento.

## Entregable II en Google Colab

Abra directamente [el Entregable II en Google Colab](https://colab.research.google.com/github/Cde571/LogisTech-Modelos-Simulacion-II/blob/main/notebooks/03_entregable_ii_completo_colab.ipynb) y use **Runtime → Run all**. El notebook descarga Olist automáticamente desde la API pública de Kaggle e instala sus dependencias; no requiere subir archivos ni escribir una clave de Kaggle.

La corrida reproducible incluye:

- cinco familias de modelos para regresión y clasificación: paramétrico, no paramétrico, ensamble, red neuronal y SVM;
- validación cruzada temporal y mallas de hiperparámetros;
- métricas de entrenamiento, validación y prueba, con intervalos de confianza;
- importancia individual de variables;
- PCA y UMAP sobre los dos mejores modelos seleccionados con validación;
- exportación de tablas a `outputs/entregable_ii/`.

Por defecto se usan 15.000 pedidos distribuidos a lo largo de todo el periodo para que la corrida termine en Colab CPU. Para utilizar todos los pedidos, cambie `MAX_ROWS = 15_000` por `MAX_ROWS = None`; esto incrementa considerablemente el tiempo de KNN, la red neuronal y UMAP.

## Estado del entregable

`report/main.tex` contiene la redacción inicial del Entregable I y marcadores explícitos para completar el estado del arte con al menos cuatro artículos reales. Antes de la entrega final, el equipo debe verificar cada fuente, completar autores e integrantes y compilar el PDF. No se incluyen referencias bibliográficas inventadas.
