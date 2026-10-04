# Datafest Coding 

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/datos.py
```

`src/datos.py` valida los contratos de train/test, normaliza las categorías y
archivos reproducibles en `salida/` sin modificar los CSV originales.
No se elimino filas ni hubo recortes de valores.
