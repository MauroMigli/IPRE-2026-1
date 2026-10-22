#!/bin/bash
#SBATCH --job-name=eeg_pipeline
#SBATCH --output=eeg_pipeline_%j.out
#SBATCH --error=eeg_pipeline_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=general

# Enviar desde la raíz del repositorio: sbatch cluster/run_pipeline.sbatch
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

# 1. Selección del orden MVAR (AIC / BIC) sobre el MVAR multi-ensayo remuestreado
python scripts/find_optimal_p.py --method mean

# 2. Análisis principal con Promedio Espacial (Mean) + TFCE
python run_pipeline.py --roi-method mean --tfce --perms 5000 --jobs 16

# 3. Análisis de robustez con PCA (1er Componente Principal)
python run_pipeline.py --roi-method pca

# 4. Chequeo de cordura: efecto del estímulo sobre la potencia espectral
python scripts/power_contrast.py --roi-method mean

# 5. Informe comparativo de robustez
python scripts/compare_robustness.py

echo "Pipeline finalizado exitosamente."
