import argparse
import os
import gc
import numpy as np
from pathlib import Path
from scipy.stats import ttest_ind
from joblib import Parallel, delayed
import time

import parameters
from src.preprocessing import get_valid_subjects
from src.connectivity import load_and_compute_ddtf
from src.statistics import compute_welch_t_map, get_spatial_adjacency_matrix, build_4d_graph, tfce_transform, fdrcorrect_bh, worker_permutation, get_3d_positions
from src.visualization import plot_edge_counts, plot_aic_bic_histograms, export_interactive_3d_network, plot_tfce_heatmaps

def run(args):
    print("==========================================================================")
    print(" IPRE-2026: Pipeline de Análisis de Conectividad EEG")
    print("==========================================================================")
    
    os.makedirs("plots", exist_ok=True)
    
    valid_subjects = get_valid_subjects()
    print(f"Sujetos válidos encontrados (ambas condiciones): {len(valid_subjects)}")
    
    D_FT = []
    D_PT = []
    global_ch_names = None
    
    bands = parameters.F_BANDS
    band_names = list(bands.keys())
    fs_global = parameters.FS_GLOBAL
    
    print("\n--- 1. Extracción y Cálculo de Conectividad dDTF ---")
    for kid_id, info in valid_subjects.items():
        ddtf_hb, ch = load_and_compute_ddtf(info['hb'], p=args.p, sampling_freq=500.0)
        if global_ch_names is None: global_ch_names = ch
        ddtf_si, _ = load_and_compute_ddtf(info['si'], p=args.p, sampling_freq=500.0)
        
        min_ep = min(len(ddtf_hb), len(ddtf_si))
        D_s = ddtf_hb[:min_ep] - ddtf_si[:min_ep]
        
        n_ep, _, n_dest, n_src = D_s.shape
        D_s_bands = np.zeros((n_ep, len(band_names), n_dest, n_src))
        
        for b_idx, b_name in enumerate(band_names):
            freq_mask = (fs_global >= bands[b_name][0]) & (fs_global <= bands[b_name][1])
            if np.any(freq_mask):
                D_s_bands[:, b_idx, :, :] = np.mean(D_s[:, freq_mask, :, :], axis=1)
                
        if info['group'] == 'FT': D_FT.append(D_s_bands)
        elif info['group'] == 'PT': D_PT.append(D_s_bands)
        
        del ddtf_hb, ddtf_si, D_s, D_s_bands
    gc.collect()
    
    global_min_epochs = min([len(D) for D in D_FT] + [len(D) for D in D_PT])
    D_FT_arr = np.array([D[:global_min_epochs] for D in D_FT])
    D_PT_arr = np.array([D[:global_min_epochs] for D in D_PT])
    
    np.save("plots/channel_names.npy", np.array(global_ch_names, dtype=object))
    
    coords_3d = get_3d_positions(parameters.ELP_FILE, global_ch_names)
    
    p_values_naive = None
    p_values_fdr = None
    p_values_tfce = None
    
    # --- 2. Naive (Welch T-Test) ---
    if args.method in ['naive', 'all', 'fdr']:
        print("\n--- 2. Calculando T-Test de Welch (Naive) ---")
        _, p_values_raw = ttest_ind(D_FT_arr, D_PT_arr, axis=0, equal_var=False, nan_policy='omit')
        p_values_naive = np.transpose(p_values_raw, (2, 3, 1, 0)) # (dest, src, band, epoch)
        np.save("plots/p_values_naive.npy", p_values_naive)
        
        export_interactive_3d_network(coords_3d, p_values_naive[:, :, 0, 0], global_ch_names, filename="plots/red_naive_b0_e0.html", dropped_channels=parameters.DROPPED_CHANNELS)
        
    # --- 3. FDR (Benjamini-Hochberg) ---
    if args.method in ['fdr', 'all']:
        print("\n--- 3. Corrección FDR (Benjamini-Hochberg) ---")
        p_values_fdr = np.zeros_like(p_values_naive)
        for e in range(p_values_naive.shape[3]):
            for b in range(p_values_naive.shape[2]):
                p_slice = p_values_naive[:, :, b, e].copy()
                np.fill_diagonal(p_slice, np.nan)
                
                valid_mask = ~np.isnan(p_slice)
                p_valid = p_slice[valid_mask]
                
                q_valid = fdrcorrect_bh(p_valid)
                
                q_slice = np.full_like(p_slice, np.nan)
                q_slice[valid_mask] = q_valid
                p_values_fdr[:, :, b, e] = q_slice
                
        np.save("plots/p_values_fdr.npy", p_values_fdr)
        export_interactive_3d_network(coords_3d, p_values_fdr[:, :, 0, 0], global_ch_names, filename="plots/red_fdr_b0_e0.html", dropped_channels=parameters.DROPPED_CHANNELS)

    # --- 4. TFCE (Permutaciones de Monte Carlo) ---
    if args.method in ['tfce', 'all']:
        print("\n--- 4. Corrección TFCE (Monte Carlo) ---")
        t_map_raw = compute_welch_t_map(D_FT_arr, D_PT_arr)
        T_map_real_4d = np.transpose(t_map_raw, (2, 3, 1, 0))
        
        print(f"  * Construyendo grafo espacial con R={args.R} cm")
        adj_spatial = get_spatial_adjacency_matrix(global_ch_names, parameters.ELP_FILE, args.R)
        adj_4d = build_4d_graph(adj_spatial, n_bands=len(band_names), n_epochs=global_min_epochs)
        
        print("  * Transformando mapa T real a TFCE")
        tfce_real = tfce_transform(T_map_real_4d, spatial_adjacency=adj_4d, dh=args.dh)
        
        print(f"  * Iniciando {args.perms} permutaciones en paralelo (Jobs={args.jobs})")
        D_all = np.concatenate([D_FT_arr, D_PT_arr], axis=0)
        N_FT = len(D_FT_arr)
        N_total = len(D_all)
        
        timer = time.time()
        max_tfce_null = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(worker_permutation)(seed, D_all, N_FT, N_total, adj_4d, args.dh)
            for seed in range(args.perms)
        )
        print(f"  * Tiempo simulaciones: {time.time() - timer:.2f} s")
        
        max_tfce_null = np.array(max_tfce_null)
        p_values_tfce = np.ones_like(tfce_real)
        
        # Calcular p-valores pseudo-empíricos
        for e in range(tfce_real.shape[3]):
            for b in range(tfce_real.shape[2]):
                for i in range(tfce_real.shape[0]):
                    for j in range(tfce_real.shape[1]):
                        if i == j: 
                            p_values_tfce[i, j, b, e] = np.nan
                        else:
                            val = tfce_real[i, j, b, e]
                            p_values_tfce[i, j, b, e] = np.sum(max_tfce_null >= val) / args.perms
                            
        np.save("plots/p_values_tfce.npy", p_values_tfce)
        export_interactive_3d_network(coords_3d, p_values_tfce[:, :, 0, 0], global_ch_names, filename="plots/red_tfce_b0_e0.html", dropped_channels=parameters.DROPPED_CHANNELS)
        
        # Plot Heatmaps (Ej: R=0, 6.44, 19.32)
        # Promediamos temporalmente el TFCE crudo para los plots de energía
        tfce_temp_avg = np.mean(tfce_real[:, :, 0, :], axis=2) # Banda 0
        plot_tfce_heatmaps(tfce_temp_avg, band_names[0], args.R, output_dir="plots")
        
    # --- 5. Gráfico Final de Tendencias (Si se corrieron todos) ---
    if args.method == 'all':
        print("\n--- 5. Generando visualizaciones consolidadas ---")
        expected_fp = (len(global_ch_names) * (len(global_ch_names) - 1)) * 0.05
        epochs_x = np.arange(global_min_epochs)
        
        for b_idx, b_name in enumerate(band_names):
            naive_counts = [np.nansum(p_values_naive[:, :, b_idx, e] < 0.05) for e in epochs_x]
            fdr_counts = [np.nansum(p_values_fdr[:, :, b_idx, e] < 0.05) for e in epochs_x]
            tfce_counts = [np.nansum(p_values_tfce[:, :, b_idx, e] < 0.05) for e in epochs_x]
            
            plot_edge_counts(epochs_x, naive_counts, fdr_counts, tfce_counts, expected_fp, b_name)
            
    print("\n¡PIPELINE COMPLETADO EXITOSAMENTE!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Conectividad EEG")
    parser.add_argument("--method", type=str, choices=['naive', 'fdr', 'tfce', 'all'], default='all', help="Método estadístico a correr")
    parser.add_argument("--p", type=int, default=7, help="Orden MVAR óptimo")
    parser.add_argument("--dh", type=float, default=0.1, help="Paso discreto dh para la integral TFCE")
    parser.add_argument("--R", type=float, default=6.44, help="Radio espacial (cm) para adyacencia TFCE")
    parser.add_argument("--perms", type=int, default=1000, help="Número de permutaciones Monte Carlo")
    parser.add_argument("--jobs", type=int, default=-1, help="Número de cores para paralelizacion (-1 = todos)")
    
    args = parser.parse_args()
    run(args)
