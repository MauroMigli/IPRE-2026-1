# Capa de Presentación (Notebooks)

Esta carpeta está destinada exclusivamente para alojar Jupyter Notebooks (`.ipynb`) utilizados para la generación de figuras y exploración interactiva de los resultados.

De acuerdo a las mejores prácticas arquitectónicas, la lógica pesada (ajuste de modelos MVAR, TFCE, FDR) se ejecuta mediante `run_pipeline.py` en el clúster, y guarda sus salidas ligeras en la carpeta `../plots/`. 

Estos notebooks deben limitarse a **leer** los archivos generados en `../plots/` o hacer uso de las funciones de `../src/visualization.py` para visualizar interactivamente:
1. Histogramas de Selección MVAR (AIC/BIC).
2. Tendencias de significancia (Naive vs E[FP]).
3. Visualizaciones 3D HTML del cerebro.
4. Mapas de Calor (Energía TFCE).
