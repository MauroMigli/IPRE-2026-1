#!/usr/bin/env python3
"""
compare_robustness.py - Análisis cuantitativo y gráfico de robustez metodológica:
Compara el impacto de la agregación espacial intra-ROI (Promedio Espacial vs. 1er Componente PCA).
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import parameters
from src.visualization import plot_robustness_comparison

def main():
    parser = argparse.ArgumentParser(
        description="Comparación de robustez metodológica: Promedio Espacial vs PCA (1er componente)"
    )
    parser.add_argument(
        "--mean-file",
        type=str,
        default="plots/p_values_rois_mean_naive.npy",
        help="Archivo p-valores con agregación mean (default: plots/p_values_rois_mean_naive.npy)"
    )
    parser.add_argument(
        "--pca-file",
        type=str,
        default="plots/p_values_rois_pca_naive.npy",
        help="Archivo p-valores con agregación pca (default: plots/p_values_rois_pca_naive.npy)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directorio donde guardar el gráfico comparativo (default: plots)"
    )
    
    args = parser.parse_args()
    
    # Si mean-file no tiene sufijo pero existe la versión base
    if not os.path.exists(args.mean_file) and os.path.exists("plots/p_values_rois_naive.npy"):
        args.mean_file = "plots/p_values_rois_naive.npy"
        
    if not os.path.exists(args.mean_file):
        print(f"[ERROR] No se encontró el archivo de resultados para 'mean': {args.mean_file}")
        print("Ejecuta primero: python run_pipeline.py --roi-method mean")
        sys.exit(1)
        
    if not os.path.exists(args.pca_file):
        print(f"\n[AVISO] Aún no se han generado los resultados para 'pca' ({args.pca_file}).")
        print("Para generarlos, ejecuta en tu cluster o entorno de cómputo:")
        print("    python run_pipeline.py --roi-method pca")
        print("\nUna vez concluido, vuelve a ejecutar este script para obtener el análisis comparativo completo.")
        sys.exit(0)
        
    p_mean = np.load(args.mean_file)
    p_pca = np.load(args.pca_file)
    
    band_names = list(parameters.F_BANDS.keys())
    n_epochs = min(p_mean.shape[3], p_pca.shape[3])
    
    print("\n==========================================================================")
    print(" ANÁLISIS DE ROBUSTEZ METODOLÓGICA (MEAN vs PCA)")
    print("==========================================================================")
    print(f"Dimensiones de matrices: {p_mean.shape} | Épocas evaluadas: {n_epochs}")
    print("\nCorrelación de Pearson entre matrices de p-valores (Mean vs PCA):")
    print("--------------------------------------------------------------------------")
    for b_idx, b_name in enumerate(band_names):
        corr_list = []
        for e in range(n_epochs):
            m_slice = p_mean[:, :, b_idx, e].flatten()
            p_slice = p_pca[:, :, b_idx, e].flatten()
            valid = ~np.isnan(m_slice) & ~np.isnan(p_slice)
            if np.sum(valid) > 2:
                r = np.corrcoef(m_slice[valid], p_slice[valid])[0, 1]
                corr_list.append(r)
        mean_r = np.nanmean(corr_list)
        print(f"  * Banda {b_name:7s}: r promedio = {mean_r:.4f}")
        
    print("--------------------------------------------------------------------------")
    
    out_img = plot_robustness_comparison(
        p_file_mean=args.mean_file,
        p_file_pca=args.pca_file,
        band_names=band_names,
        output_dir=args.output_dir
    )
    if out_img:
        print(f"\n[ÉXITO] Gráfico de robustez generado en: {out_img}")

if __name__ == "__main__":
    main()
