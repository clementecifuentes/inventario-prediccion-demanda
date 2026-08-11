"""
Perfilado del dataset antes de analizarlo.

Chequeos de integridad (nulos, duplicados, cobertura del calendario) y dos
pruebas que evalúan si la demanda tiene la irregularidad propia de datos
medidos o el comportamiento demasiado regular de datos generados.

Uso:
    python src/perfilado_datos.py
"""

import sys

import pandas as pd

# La consola de Windows usa cp1252 por defecto y rompe con los acentos
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def cargar() -> pd.DataFrame:
    df = pd.read_csv("data/ventas.csv", parse_dates=["date"])
    return df.rename(columns={"date": "fecha", "store": "tienda",
                              "item": "sku", "sales": "unidades"})


def integridad(df: pd.DataFrame) -> None:
    print("== Integridad ==")
    print(f"filas: {len(df):,}")
    print(f"nulos: {df.isna().sum().sum()}")
    print(f"duplicados (tienda-sku-fecha): "
          f"{df.duplicated(['tienda', 'sku', 'fecha']).sum()}")
    print(f"unidades negativas: {(df['unidades'] < 0).sum()}")

    dias = df["fecha"].nunique()
    esperados = (df["fecha"].max() - df["fecha"].min()).days + 1
    print(f"días con datos: {dias:,} de {esperados:,} "
          f"({'sin huecos' if dias == esperados else 'HAY HUECOS'})")

    combinaciones = df.groupby(["tienda", "sku"]).size()
    print(f"series tienda-sku: {len(combinaciones)} "
          f"(cada una con {combinaciones.min():,}-{combinaciones.max():,} días)")


def variabilidad(df: pd.DataFrame) -> None:
    """
    En datos reales el coeficiente de variación difiere mucho entre productos:
    conviven artículos de demanda estable con otros erráticos.
    """
    print("\n== Variabilidad entre productos ==")
    diario = df.groupby(["sku", "fecha"])["unidades"].sum()
    stats = diario.groupby("sku").agg(["mean", "std"])
    cv = stats["std"] / stats["mean"]

    print(f"CV mínimo: {cv.min():.4f}")
    print(f"CV máximo: {cv.max():.4f}")
    print(f"amplitud:  {cv.max() - cv.min():.4f}")
    print(f"volumen del SKU mayor / SKU menor: "
          f"{stats['mean'].max() / stats['mean'].min():.1f}x")

    if cv.max() - cv.min() < 0.05:
        print("-> Los 50 productos tienen prácticamente el mismo CV pese a "
              "diferencias de volumen de casi 5x. En datos medidos esto no "
              "ocurre.")


def proporciones(df: pd.DataFrame) -> None:
    """
    En datos reales cada tienda tiene su propio mix: vende más de unos productos
    y menos de otros. Una proporción constante indica generación multiplicativa.
    """
    print("\n== Mix de producto por tienda ==")
    piv = df.groupby(["tienda", "sku"])["unidades"].sum().unstack()
    part = piv.div(piv.sum(axis=0), axis=1)   # participación de cada tienda por SKU
    dispersion = part.std(axis=1).mean()

    print(f"participación de cada tienda: "
          f"{part.mean(axis=1).min():.1%} a {part.mean(axis=1).max():.1%}")
    print(f"desvío de esa participación entre productos: {dispersion:.5f}")

    if dispersion < 0.005:
        print("-> Cada tienda mantiene la misma cuota en los 50 productos. "
              "No hay preferencias locales, algo que ninguna cadena real "
              "presenta.")


def main() -> None:
    df = cargar()
    integridad(df)
    variabilidad(df)
    proporciones(df)
    print("\nConclusión: la serie es apta para practicar forecasting "
          "(estacionalidad y tendencia bien formadas), pero las conclusiones "
          "de negocio no son extrapolables a una operación real.")


if __name__ == "__main__":
    main()
