#!/usr/bin/env python3
"""
find_optimal_p.py - Búsqueda formal y reproducible del orden óptimo de MVAR (p)
usando Criterios de Información de Akaike (AIC) y Bayesiano (BIC) sobre Super-Nodos (ROIs),
con un MVAR multi-ensayo por registro sobre señales remuestreadas a parameters.FS_TARGET.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import parameters
from src.preprocessing import get_valid_subjects
from src.model_order import find_dataset_optimal_order, evaluate_orders
from src.visualization import plot_order_selection_curves

def run_synthetic_benchmark(max_p=15, n_epochs=30, n_times=150, n_rois=8, method="mean"):
    """
    Ejecuta una evaluación simulada con un MVAR multi-ensayo de orden conocido
    para verificar el módulo cuando no hay datos crudos locales.
    """
    print(f"\n[BENCHMARK SINTÉTICO] Generando {n_epochs} épocas de prueba ({n_rois} ROIs, {n_times} muestras, método: {method})...")
    rng = np.random.default_rng(42 if method == "mean" else 100)
    p_true = 5
    
    signal = np.zeros((n_epochs, n_rois, n_times))
    for ep in range(n_epochs):
        noise = rng.standard_normal((n_rois, n_times))
        for t in range(p_true, n_times):
            signal[ep, :, t] = 0.3 * signal[ep, :, t-1] - 0.2 * signal[ep, :, t-p_true] + noise[:, t]
            
    res = evaluate_orders(signal, max_p=max_p)
    lags = res['lags']
    plot_order_selection_curves(
        lags, res['aic'], res['bic'], [lags[np.argmin(res['aic'])]], [lags[np.argmin(res['bic'])]],
        output_dir="plots", 
        suffix=f"_{method}_synthetic",
        method_label=f"Sintético: {method.upper()}"
    )
    opt_p = lags[np.argmin(res['bic'])]
    print(f"[BENCHMARK SINTÉTICO] Orden óptimo detectado por BIC: p = {opt_p} (Esperado {p_true})")

def main():
    parser = argparse.ArgumentParser(
        description="Búsqueda reproducible del orden MVAR óptimo (p) vía AIC/BIC sobre Super-Nodos (ROIs)"
    )
    parser.add_argument("--max-p", type=int, default=parameters.MAX_P, help="Rezago máximo a evaluar (default: parameters.MAX_P)")
    parser.add_argument("--method", type=str, choices=['mean', 'pca'], default='mean', 
                        help="Método de agregación de canales a ROI: 'mean' o 'pca' (default: 'mean')")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directorio para guardar gráficos y JSON")
    parser.add_argument("--synthetic", action="store_true", help="Correr prueba sintética si no hay datos crudos")
    
    args = parser.parse_args()
    
    valid_subjects = get_valid_subjects()
    if len(valid_subjects) == 0:
        print("\n[AVISO] No se encontraron sujetos con datos en data/epch_heartbeat y data/epch_silence.")
        if args.synthetic:
            run_synthetic_benchmark(max_p=args.max_p, method=args.method)
            return
        else:
            print("Para probar la funcionalidad con datos simulados, usa: python scripts/find_optimal_p.py --synthetic")
            sys.exit(0)
            
    results = find_dataset_optimal_order(
        valid_subjects,
        rois_dict=parameters.ROIS,
        max_p=args.max_p,
        method=args.method,
        output_dir=args.output_dir
    )
    
    if results is not None:
        plot_order_selection_curves(
            np.array(results['lags']),
            np.array(results['mean_aic']),
            np.array(results['mean_bic']),
            np.array(results['aic_votes']),
            np.array(results['bic_votes']),
            output_dir=args.output_dir,
            suffix=f"_{args.method}",
            method_label=f"ROI: {args.method.upper()}"
        )
        print(f"\n[INFO] Gráfico guardado en: {args.output_dir}/mvar_order_selection_curves_{args.method}.png")
        print(f"[RECOMENDACIÓN] Puedes configurar 'P_OPTIMO = {results['recommended_p']}' en parameters.py")
        print(f"                o ejecutar el pipeline con: python run_pipeline.py --roi-method {args.method} --p {results['recommended_p']}")

if __name__ == "__main__":
    main()
