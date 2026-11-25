# Seminario de Tesis II · Predicción de eventos climáticos extremos en Puno

Modelo de Machine Learning basado en series temporales para la predicción de eventos climáticos extremos en la región de Puno con datos del SENAMHI.
Autor: Antonio Arthur Quispe Javier (UNI, Maestría en Ciencias en Ciencia de la Computación). Asesora: Ph.D. Rosa Virginia Encinas Quille.

## Contenido
- `cuadernos/`: tres cuadernos de Google Colab (01 limpieza, 02 descarga ERA5, 03 modelos, evaluación y exportación).
- `codigo/`: los mismos módulos en Python (`common.py`, `run_xgb.py`, `dl.py`, `evaluate.py`, `figures.py`, `export_powerbi.py`).
- `resultados/`: métricas del periodo de prueba (2018–2026), cobertura de datos y predicciones con niveles de alerta.
- `tablero_web/`: prototipo web del tablero (abrir `index.html` con un servidor local, p. ej. `python -m http.server`).
- `Guia_PowerBI.md`: cómo cargar el CSV en Power BI Desktop.

## Datos
Los archivos originales del SENAMHI (`.xlsx`) **no se incluyen** (uso académico bajo declaración jurada). Para reproducir, colóquelos en `data/` y ejecute:

```
python codigo/run_xgb.py data out
python codigo/dl.py data out lstm 42,43,44,45,46
python codigo/dl.py data out tft 42,43,44
python codigo/evaluate.py out && python codigo/figures.py data out && python codigo/export_powerbi.py out
```
Los scripts importan `common.py`, por lo que deben ejecutarse desde la carpeta `codigo/` o con ella en `PYTHONPATH`.

## Resumen de resultados (prueba 2018–2026, horizonte de 1 día)
Temperatura media: R² ≈ 0,84 (RMSE ≈ 1,08 °C) con XGBoost, LSTM y TFT. Precipitación: R² ≈ 0,23. Día frío extremo: F1 ≈ 0,43. Lluvia intensa: F1 ≈ 0,09–0,11. Detalles en `resultados/`.

Limitaciones: sin temperatura mínima/máxima (las heladas no se modelan directamente), TFT simplificado sin variables externas, eventos raros.
