# Guía rápida: tablero en Power BI Desktop
1. Obtener datos > Texto/CSV > `resultados/powerbi_predicciones_alertas.csv` (codificación UTF-8). Convertir `fecha`/`fecha_objetivo` a Fecha.
2. Filtro de segmentación: `periodo` = prueba; segmentadores: `estacion`, `fecha_objetivo`.
3. Tarjetas: días en Alerta (medida `Alertas Lluvia = CALCULATE(COUNTROWS(T), T[nivel_lluvia]="Alerta")`, igual para `nivel_frio`).
4. Gráfico de líneas: `TM_obs_t+1` vs `TM_pred_xgb` por fecha; columnas: `PT_obs_t+1` y `PT_pred_xgb` con línea constante `umbral_lluvia_mm`.
5. Matriz: filas `nivel_frio`, columna `evento_frio_obs`, valores = Recuento (aciertos/falsas alarmas).
6. Mapa: agregar latitud/longitud de estaciones cuando el SENAMHI las entregue.
