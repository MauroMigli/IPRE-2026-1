#!/usr/bin/env python3
"""
find_optimal_p.py - Búsqueda formal y reproducible del orden óptimo de MVAR (p)
usando Criterios de Información de Akaike (AIC) y Bayesiano (BIC) sobre Super-Nodos (ROIs).
"""

import argparse
import os
import sys
import numpy as np

import parameters
from src.preprocessing import get_valid_subjects
from src.model_order import find_dataset_optimal_order, evaluate_epoch_orders
from src.visualization import plot_order_selection_curves

def run_synthetic_benchmark(max_p=15, n_epochs=20, n_times=250, n_rois=8):
    """
    Ejecuta una evaluación simulada con señales multivariadas de prueba
    para verificar la estabilidad del módulo cuando no hay datos crudos locales.
    """
    print(f"\n[BENCHMARK SINTÉTICO] Generando {n_epochs} épocas de prueba ({n_rois} ROIs, {n_times} muestras)...")
    np.random.seed(42)
    # Generar proceso autorregresivo multivariado sintético con p_true = 5
    p_true = 5
    lags = np.arange(1, max_p + 1)
    
    all_aic = []
    all_bic = []
    
    for _ in range(n_epochs):
        noise = np.random.randn(n_times, n_rois)
        signal = np.zeros_like(noise)
        for t in range(p_true, n_times):
            ar_term = 0.3 * signal[t-1] - 0.2 * signal[t-p_true]
            signal[t] = ar_term + noise[t]
            
        res = evaluate_epoch_orders(signal, max_p=max_p)
        all_aic.append(res['aic'])
        all_bic.append(res['bic'])
        
    all_aic = np.vstack(all_aic)
    all_bic = np.vstack(all_bic)
    
    mean_aic = np.nanmean(all_aic, axis=0)
    mean_bic = np.nanmean(all_bic, axis=0)
    aic_votes = lags[np.nanargmin(all_aic, axis=1)]
    bic_votes = lags[np.nanargmin(all_bic, axis=1)]
    
    plot_order_selection_curves(lags, mean_aic, mean_bic, aic_votes, bic_votes, output_dir="plots")
    opt_p = lags[np.nanargmin(mean_bic)]
    print(f"[BENCHMARK SINTÉTICO] Orden óptimo detectado por BIC: p = {opt_p} (Esperado cercano a {p_true})")
    print("Gráfico generado en: plots/mvar_order_selection_curves.png")

def main():
    parser = argparse.ArgumentParser(
        description="Búsqueda reproducible del orden MVAR óptimo (p) vía AIC/BIC sobre Super-Nodos (ROIs)"
    )
    parser.add_argument("--max-p", type=int, default=15, help="Rezago máximo a evaluar (default: 15)")
    parser.add_argument("--method", type=str, choices=['mean', 'pca'], default='mean', 
                        help="Método de agregación de canales a ROI: 'mean' o 'pca' (default: 'mean')")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directorio para guardar gráficos y JSON")
    parser.add_argument("--synthetic", action="store_true", help="Correr prueba sintética si no hay datos crudos")
    
    args = parser.parse_args()
    
    valid_subjects = get_valid_subjects()
    if len(valid_subjects) == 0:
        print("\n[AVISO] No se encontraron sujetos con datos en data/epch_heartbeat y data/epch_silence.")
        if args.synthetic:
            run_synthetic_benchmark(max_p=args.max_p)
            return
        else:
            print("Para probar la funcionalidad con datos simulados, usa: python find_optimal_p.py --synthetic")
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
            output_dir=args.output_dir
        )
        print(f"\n[INFO] Gráfico comparativo guardado en: {args.output_dir}/mvar_order_selection_curves.png")
        print(f"[RECOMENDACIÓN] Puedes configurar 'P_OPTIMO = {results['recommended_p']}' en parameters.py")
        print(f"                o ejecutar el pipeline con: python run_pipeline.py --p {results['recommended_p']}")

if __name__ == "__main__":
    main()
