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

# Ejecutar el orquestador principal
echo "Iniciando ejecución del pipeline EEG..."
python ../run_pipeline.py --method all --R 6.44 --perms 1000 --jobs 16
echo "Pipeline finalizado exitosamente."
