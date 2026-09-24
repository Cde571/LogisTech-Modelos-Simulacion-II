# Datos

Los datos originales y procesados no se suben al repositorio. Descargue el dataset [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) y extraiga los archivos en `data/raw/`.

Comando recomendado:

```bash
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip
```

## Archivos usados

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_products_dataset.csv`
- `olist_order_payments_dataset.csv`

El cargador también reconoce, si están presentes:

- `olist_sellers_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_geolocation_dataset.csv`
- `product_category_name_translation.csv`

## Directorios

- `raw/`: CSV originales sin modificar.
- `processed/`: tablas derivadas generadas localmente.

Ambos directorios están ignorados por Git salvo sus archivos `.gitkeep`. No incluya credenciales de Kaggle, datos descargados ni artefactos de modelos en commits.

## Definición de objetivos

- `delivery_time_days = order_delivered_customer_date - order_purchase_timestamp`, expresado en días.
- `late_delivery = 1` si `order_delivered_customer_date > order_estimated_delivery_date`; en otro caso vale 0.

Solo se conservan pedidos con entrega observada para construir estos objetivos. Las columnas posteriores a la compra se eliminan del conjunto de predictores.
