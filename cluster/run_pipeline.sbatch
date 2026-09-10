#!/bin/bash
#SBATCH --job-name=eeg_pipeline
#SBATCH --output=eeg_pipeline_%j.out
#SBATCH --error=eeg_pipeline_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=general

# Activar entorno virtual
source ../.venv/bin/activate

# 1. (Opcional) Búsqueda previa del orden óptimo MVAR (AIC / BIC) para ambos métodos
# python ../scripts/find_optimal_p.py --max-p 15 --method mean
# python ../scripts/find_optimal_p.py --max-p 15 --method pca

# 2. Ejecutar análisis principal con Promedio Espacial (Mean)
echo "Ejecutando pipeline con Promedio Espacial (Mean)..."
python ../run_pipeline.py --method fdr --use-rois --roi-method mean

# 3. Ejecutar análisis de robustez con PCA (1er Componente Principal)
echo "Ejecutando pipeline de robustez con PCA (PC1)..."
python ../run_pipeline.py --method fdr --use-rois --roi-method pca

# 4. Generar visualizaciones interactivas 3D temporales (Delta y Gamma)
python ../scripts/plot_temporal_network.py --band both --roi-method mean
python ../scripts/plot_temporal_network.py --band both --roi-method pca

# 5. Generar informe comparativo de robustez cuantitativo
python ../scripts/compare_robustness.py

echo "Pipeline de robustez finalizado exitosamente."
