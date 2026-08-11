"""
Descarga el dataset "Store Item Demand Forecasting Challenge" (Kaggle):
ventas diarias de 50 productos en 10 tiendas, 2013-2017 (~913.000 filas).

Fuente original: https://www.kaggle.com/c/demand-forecasting-kernels-only
Se descarga desde un espejo público en GitHub para que el pipeline sea
reproducible sin credenciales de Kaggle.

Uso:
    python src/descargar_datos.py
"""

import sys

# La consola de Windows usa cp1252 y rompe con acentos y flechas
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import requests

URL = ("https://raw.githubusercontent.com/allmeidaapedro/"
       "Store-Item-Demand-Forecasting/main/input/train.csv")


def descargar(destino: str = "data/ventas.csv") -> None:
    ruta = Path(destino)
    ruta.parent.mkdir(exist_ok=True)
    if ruta.exists():
        print(f"{ruta} ya existe, se omite")
        return
    print("descargando ventas diarias (~17 MB)...")
    r = requests.get(URL, timeout=300)
    r.raise_for_status()
    ruta.write_bytes(r.content)
    print(f"Listo: {ruta} ({len(r.content) / 1e6:.1f} MB)")


if __name__ == "__main__":
    try:
        descargar()
    except requests.RequestException as exc:
        sys.exit(f"Error de descarga: {exc}")
