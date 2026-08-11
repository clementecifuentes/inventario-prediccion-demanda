# Inventario y predicción de demanda 📦

Gestión de inventario basada en datos sobre **913.000 registros de ventas diarias**
(50 productos × 10 tiendas, 2013-2017): análisis ABC, matriz volumen-variabilidad,
pronóstico de demanda con Holt-Winters y política de reposición con punto de
reorden y stock de seguridad.

**Stack:** Python · Pandas · NumPy · statsmodels · Matplotlib

---

## Hallazgos principales

- El **análisis ABC** muestra una concentración moderada: 31 de 50 SKUs explican
  el 80% del volumen. No hay una cola larga extrema — toda la cartera merece
  seguimiento, pero las clases B y C admiten políticas de reposición más simples.
- La **variabilidad relativa de la demanda es homogénea** (~25% de coeficiente de
  variación) y decrece suavemente con el volumen — el patrón esperable de demanda
  aleatoria estable. Eso permite una política de stock de seguridad uniforme por
  fórmula, sin tratamiento especial por SKU.
- El pronóstico mensual con **Holt-Winters logra 2,4% de MAPE** sobre 2017
  (validación fuera de muestra), superando al baseline naive estacional (3,4%).
  La variante de estacionalidad aditiva se seleccionó por validación — la
  multiplicativa rendía peor (4,4%).
- La **política de reposición** calculada (lead time 7 días, 95% de servicio)
  arroja puntos de reorden de 6.200-7.100 unidades para los SKUs líderes, con el
  stock de seguridad representando ~13% del punto de reorden.

## Visualizaciones

### Clasificación ABC
![Análisis ABC](figures/01_abc.png)

### Perfil de demanda por producto
![Matriz volumen-variabilidad](figures/02_matriz_volumen_variabilidad.png)

### Pronóstico de demanda
![Pronóstico Holt-Winters](figures/03_pronostico.png)

### Política de reposición
![Punto de reorden y stock de seguridad](figures/04_politica_inventario.png)

## Fuente de datos

[Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only)
(Kaggle) — ventas diarias de 50 productos en 10 tiendas durante 5 años.
Se descarga desde un espejo público para que el pipeline sea reproducible
sin credenciales.

## Reproducir el análisis

```bash
pip install -r requirements.txt

# 1. Descargar los datos (~17 MB)
python src/descargar_datos.py

# 2. Generar las figuras
python src/analisis.py
```

## Estructura

```
├── src/
│   ├── descargar_datos.py   # descarga del dataset
│   └── analisis.py          # ABC, forecast y política de inventario
├── figures/                 # gráficos generados (PNG)
├── data/                    # datos crudos (no versionados)
└── requirements.txt
```

## Notas metodológicas

- **ABC**: clase A hasta el 80% del volumen acumulado, B hasta el 95%, C el resto.
- **Pronóstico**: entrenamiento 2013-2016, evaluación sobre 2017 completo.
  Modelo `ExponentialSmoothing(trend="add", seasonal="add", seasonal_periods=12)`.
- **Punto de reorden** = demanda media diaria × lead time + stock de seguridad,
  con SS = z·σ·√LT (z = 1,65 para 95% de nivel de servicio, LT = 7 días).
- Los parámetros de lead time y nivel de servicio son supuestos configurables
  en `src/analisis.py`.

---

**Clemente Cifuentes** — Data Analyst ·
[LinkedIn](https://linkedin.com/in/clementecifuentes) ·
[Portafolio](https://github.com/clementecifuentes)
