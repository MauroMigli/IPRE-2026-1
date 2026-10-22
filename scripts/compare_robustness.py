#!/usr/bin/env python3
"""
compare_robustness.py - Análisis cuantitativo y gráfico de robustez metodológica:
Compara el impacto de la agregación espacial intra-ROI (Promedio Espacial vs. 1er Componente PCA).
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import parameters
from src.visualization import plot_robustness_comparison

def main():
    parser = argparse.ArgumentParser(
        description="Comparación de robustez metodológica: Promedio Espacial vs PCA (1er componente)"
    )
    parser.add_argument("--mean-file", type=str, default="plots/t_values_group_mean.npy",
                        help="Mapa t FT vs PT con agregación mean (default: plots/t_values_group_mean.npy)")
    parser.add_argument("--pca-file", type=str, default="plots/t_values_group_pca.npy",
                        help="Mapa t FT vs PT con agregación pca (default: plots/t_values_group_pca.npy)")
    parser.add_argument("--output-dir", type=str, default="plots",
                        help="Directorio donde guardar el gráfico comparativo (default: plots)")
    args = parser.parse_args()
    
    for path, method in ((args.mean_file, 'mean'), (args.pca_file, 'pca')):
        if not os.path.exists(path):
            print(f"[ERROR] No se encontró {path}. Ejecuta primero: python run_pipeline.py --roi-method {method}")
            sys.exit(1)
            
    band_names = list(parameters.F_BANDS.keys())
    out_img, correlations = plot_robustness_comparison(
        t_file_mean=args.mean_file,
        t_file_pca=args.pca_file,
        band_names=band_names,
        output_dir=args.output_dir
    )
    
    print("\n==========================================================================")
    print(" ANÁLISIS DE ROBUSTEZ METODOLÓGICA (MEAN vs PCA)")
    print(" Correlación de Pearson entre los mapas t FT vs PT (aristas fuera de la diagonal):")
    print("==========================================================================")
    for b_name, r in correlations.items():
        print(f"  * Banda {b_name:7s}: r = {r:.4f}")
    print(f"\n[ÉXITO] Gráfico de robustez generado en: {out_img}")

if __name__ == "__main__":
    main()
