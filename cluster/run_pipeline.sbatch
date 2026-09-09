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

# 1. (Opcional) Búsqueda previa del orden óptimo MVAR (AIC / BIC)
# python ../find_optimal_p.py --max-p 15 --method mean

# 2. Ejecutar el orquestador principal sobre super-nodos (ROIs)
echo "Iniciando ejecución del pipeline EEG (Super-Nodos ROIs)..."
python ../run_pipeline.py --method all --use-rois --perms 1000 --jobs 16
echo "Pipeline finalizado exitosamente."
