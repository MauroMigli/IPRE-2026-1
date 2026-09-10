# Archivo de Resultados y Metodologías Legadas (Legacy Archive)

Este directorio preserva de manera estructurada los resultados, gráficos y código de las formulaciones iniciales previas a la transición hacia **Super-Nodos (ROIs)** y **MVAR Multivariado Global**.

## Contenido

### 1. `legacy_plots/`
* **`hist_aic.png` y `hist_bic.png`:** Histogramas de votación de orden $p$ provenientes del MVAR bivariado (1.035 pares por época). Muestran la dispersión caótica e inconsistencia entre pares independientes antes de unificar el orden mediante el MVAR global sobre ROIs.
* **`p_values_welch_naive.npy` y `p_values_welch_fdr.npy`:** Matrices de contraste canal-a-canal (46 canales, 124.200 pruebas de hipótesis). Muestran cómo el test Naive arrojaba 6.105 conexiones (exactamente el 5% de ruido esperado: $124.200 \times 0.05 = 6.210$), provocando que FDR resultara en 0 conexiones supervivientes ($q_{\min} \approx 0.886$).
* **`p_values_R_*.npy`:** Resultados del método TFCE sobre la formulación antigua para radios $R \in [0, 6.44]\text{ cm}$ (donde el p-valor mínimo alcanzado fue apenas $\approx 0.38 - 0.59$).
* **`evolution.gif`:** Animación de la evolución temporal de la red de 46 canales.
* **`channel_names.npy`:** Lista de 46 canales individuales limpios.

### 2. `legacy_tests/`
* Pruebas unitarias originales del pipeline antes de la consolidación de la suite en el entorno principal.

---

## Utilidad Científica para el Paper y Tesis

> [!NOTE]
> Estos datos **no deben eliminarse**, pues constituyen la evidencia empírica que justifica ante revisores científicos y comisiones de tesis por qué:
> 1. Un análisis canal-a-canal en MVAR sufre de sobreajuste e indeterminación dimensional ($T < K \cdot p$).
> 2. El MVAR bivariado par-a-par introduce sesgo de causa común y dispersión en la selección de orden.
> 3. El agrupamiento formal en 8 ROIs (Carnevali et al., 2026; Haufe et al., 2009) y el MVAR multivariado son metodológicamente estrictos e indispensables.

