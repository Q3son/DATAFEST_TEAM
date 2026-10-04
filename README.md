# Datafest Coding 
Valida, explora, crea features y entrega datos limpios + contrato al equipo ML.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/datos.py
```

`src/datos.py` valida los contratos de train/test, normaliza las categorías y
genera archivos reproducibles en `salida/` sin modificar los CSV originales.
No elimina filas ni recorta valores extremos automáticamente: los problemas de
esquema o de dominio detienen la exportación con un mensaje explícito.

## Archivos preparados para ML

- `train_limpio.csv` y `test_limpio.csv`: columnas originales en orden
  canónico, categorías normalizadas y faltantes imputados con estadísticas de
  train.
- `train_modelo.csv`: matriz numérica one-hot más `objetivo`.
- `test_modelo.csv`: matriz con las mismas features y el mismo orden que train.
- `test_ids.csv`: ID y orden original del test para reconstruir la entrega.
- `reporte_calidad.json`: conteos, validaciones y método de imputación.
- `contrato_features.json`: categorías y medianas usadas para transformar
  features.

`id_cliente` no se incluye como predictor para evitar fuga por identificador;
se conserva en los CSV limpios y en `test_ids.csv`. `mes` sí se conserva como
feature temporal. Las variables numéricas no se escalan, para que el equipo de
ML pueda elegir el preprocesamiento adecuado para el modelo final.

Se pueden pasar rutas personalizadas con `--train`, `--test`,
`--sample-submission` y `--output-dir`.
