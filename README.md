# EEG Functional Connectivity Pipeline: dDTF & TFCE

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![MNE-Python](https://img.shields.io/badge/MNE--Python-Processing-313639.svg)
![Statsmodels](https://img.shields.io/badge/statsmodels-MVAR%20Fitting-green.svg)
![HPC Ready](https://img.shields.io/badge/HPC-Slurm%20Ready-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository documents my undergraduate research project **(IPre)** conducted during the first semester of 2026 under the supervision of Professor Mircea Petrache (Faculty of Mathematics - PUC) and Professor Marcela Peña (School of Psychology - PUC). The main goal is to analyze functional connectivity in EEG data using direct Directed Transfer Function (dDTF) and evaluate statistical significance via Threshold-Free Cluster Enhancement (TFCE).

## Pipeline Architecture (Super-Nodos / ROIs)

```mermaid
graph TD
    A[Raw EEG Data .set/.fdt] -->|MNE-Python| B(Preprocessing & Channel Pruning - 46 Channels)
    B -->|Spatial Pooling / PCA| C[8 Anatomical Super-Nodes / ROIs]
    C -->|Statsmodels| D[Multivariate Global MVAR Model p]
    D -->|Fourier Transform| E[direct Directed Transfer Function dDTF]
    E -->|O1 Caching| F[(Cache Storage _ddtf_rois.npy)]
    F --> G{Statistical Contrasting FT vs PT}
    G -->|Naive| H[Welch t-test]
    G -->|FDR| I[Benjamini-Hochberg Correction]
    G -->|TFCE + Monte Carlo| J[Topological Cluster Enhancement on ROI Centroids]
    H & I & J --> K[3D Interactive Networks & Heatmaps]
```

## Model Order Selection (AIC & BIC)

Para evitar fijar el orden del rezago $p$ arbitrariamente, dispones de una herramienta reproducible y modularizada en `scripts/` para calcular los criterios de información de Akaike (AIC) y Bayesiano (BIC) sobre las épocas multivariadas de super-nodos:

```bash
# Evaluar orden óptimo p en todo el dataset y generar curvas en plots/
python scripts/find_optimal_p.py --max-p 15 --method mean
```

Esto generará `plots/mvar_order_selection_curves.png` (curvas de AIC/BIC y distribución de votos por época) y el archivo de métricas estructurado `plots/mvar_order_selection.json`.

## Temporal 3D Network Visualization

Para inspeccionar la dinámica temporal y reproducir animaciones interactivas en 3D con barra deslizante de épocas:

```bash
# Generar visualización 3D interactiva para Gamma y Delta (guardadas en plots/)
python scripts/plot_temporal_network.py --band Gamma --roi-method mean
python scripts/plot_temporal_network.py --band Delta --roi-method mean
```

## Methodological Robustness Analysis (Mean vs. PCA)

Para validar que los descubrimientos neurobiológicos (pico temprano en Gamma época 1 y Delta época 2) son invariantes al método de reducción intra-ROI, puedes contrastar el **Promedio Espacial (`mean`)** contra el **Primer Componente Principal (`pca`)**:

```bash
# 1. Evaluar orden MVAR para ambos métodos
python scripts/find_optimal_p.py --method mean
python scripts/find_optimal_p.py --method pca

# 2. Correr el pipeline con promedio espacial y con PCA (los archivos se guardan con sufijo _mean y _pca)
python run_pipeline.py --roi-method mean
python run_pipeline.py --roi-method pca

# 3. Generar el informe y gráfico comparativo de robustez
python scripts/compare_robustness.py
```

## How to Run

El pipeline está completamente automatizado y preparado para entornos locales y HPC / Slurm. La ejecución se gestiona mediante el orquestador `run_pipeline.py`.

### Basic Execution
Ejecutar el contraste estadístico estándar (Naive y FDR) sobre los 8 super-nodos (ROIs):
```bash
python run_pipeline.py
```

### Command-Line Arguments (`argparse`)
Puedes personalizar la ejecución con las siguientes banderas:

* `--use-rois`: Opera sobre los 8 super-nodos ROIs anatómicamente definidos. *(Habilitado por defecto)*
* `--no-rois`: Desactiva el modo ROIs y ejecuta el análisis canal a canal bivariado legado.
* `--roi-method`: Método de agregación de canales dentro de cada ROI: `mean` (promedio espacial) o `pca` (primer componente principal). *(Default: `mean`)*
* `--select-order`: Ejecuta la búsqueda empírica de orden MVAR vía AIC/BIC antes de iniciar el cálculo de conectividad.
* `--method`: Método estadístico a correr: `fdr` (Naive + FDR, rápido y estándar), `tfce`, o `all`. *(Default: `fdr`)*
* `--p`: Orden del modelo MVAR. *(Default: el fijado en `parameters.P_OPTIMO = 5`)*
* `--R`: Radio espacial (en cm) para adyacencia espacial en TFCE. *(Default: automático, 9.5 cm para ROIs)*
* `--dh`: Paso discreto para la integral de Riemann en TFCE. *(Default: `0.1`)*
* `--perms`: Número de permutaciones Monte Carlo para TFCE. *(Default: `1000`)*
* `--jobs`: Número de cores CPU para paralelización (-1 = todos). *(Default: `-1`)*

**Ejemplo HPC run:**
```bash
python run_pipeline.py --method all --use-rois --perms 5000 --jobs 16
```

## Adding New Data & Configuration

All global settings and data paths are centralized in the `parameters.py` file. To add new data:

1. **Place your `.set` and `.fdt` files** inside the data directories (`data/epch_heartbeat/` and `data/epch_silence/`).
2. **Update `parameters.py`** if your directory names differ:
   ```python
   HB_DIR = Path("data/your_new_hb_folder")
   SI_DIR = Path("data/your_new_si_folder")
   ```
3. Ensure the spatial coordinates file (`data/eeglab_65chanlocs.elp`) is present, as it is strictly required to calculate spatial distances for the 3D plots and TFCE clustering.
4. Customize dropped channels or frequency bands (`F_BANDS`) directly in `parameters.py` as needed.

## Output Structure
All generated outputs are automatically saved in the `plots/` directory:
- **`p_values_rois_*.npy`**: Raw matrices of statistical p-values ($5 \times 12 \times 8 \times 8$).
- **`red_temporal_*.html`**: Interactive 3D scalp plots of significant connections over time (with play/pause animation and epoch slider).
- **`edge_counts_*.png`**: Temporal profile of significant edge counts across epochs per frequency band.
- **`archive/`**: Contains legacy 46-channel bivariate outputs and exploratory plots (`archive/legacy_plots/`) and legacy unit tests (`archive/legacy_tests/`).

## Figures
<br>
<div align="center">
  <img src="plots/edge_counts_Gamma.png" alt="Temporal Evolution of Connectivity - Gamma Band" width="75%" />
  <p>
    <br>
    <em><b>Figure 1:</b> Temporal profile of significant directed edges in the <b>Gamma band</b> across the 12 experimental epochs under FDR and Naive thresholds. A marked burst in functional connectivity is observed in Early Evoked Epoch 1 (19 surviving connections under uncorrected contrast, 1 surviving under strict FDR), returning to basal levels in subsequent epochs.</em>
  </p>
</div>
<br>
<div align="center">
  <img src="plots/mvar_order_selection_curves.png" alt="MVAR Order Selection AIC/BIC Curves" width="75%" />
  <p>
    <br>
    <em><b>Figure 2:</b> Empirical MVAR model order selection across subjects and epochs using Akaike (AIC) and Bayesian (BIC) information criteria over the 8 ROIs, justifying the selection of $p=5$.</em>
  </p>
</div>
<br>
<div align="center">
  <img src="data/mapaElectrodes.jpg" alt="Map of electrodes" width="75%" />
  <p>
    <br>
    <em><b>Figure 2:</b> Map of channels. </em>
  </p>
</div>
