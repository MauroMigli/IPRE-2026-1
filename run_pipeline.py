import argparse
import os
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp
try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel, delayed = None, None
import time

import parameters
from src.preprocessing import get_valid_subjects
from src.connectivity import load_and_compute_ddtf
from src.statistics import compute_welch_t_map, get_spatial_adjacency_matrix, build_edge_band_graph, tfce_transform, fdrcorrect_bh, worker_permutation, get_roi_3d_centroids
from src.visualization import export_interactive_3d_network, plot_contrast_heatmaps, plot_group_condition_means, plot_order_selection_curves, plot_robustness_comparison
from src.model_order import find_dataset_optimal_order


def band_average(ddtf):
    """(n_fs, dest, src) -> (n_bands, dest, src) promediando dentro de cada banda."""
    fs = parameters.FS_GLOBAL
    out = np.zeros((len(parameters.F_BANDS),) + ddtf.shape[1:])
    for b_idx, (f_lo, f_hi) in enumerate(parameters.F_BANDS.values()):
        out[b_idx] = ddtf[(fs >= f_lo) & (fs <= f_hi)].mean(axis=0)
    return out


def fdr_over_edges(p_map):
    """FDR (BH) en UNA sola familia: todas las aristas fuera de la diagonal y todas las bandas."""
    off_diag = ~np.eye(p_map.shape[0], dtype=bool)[:, :, None].repeat(p_map.shape[2], axis=2)
    q_map = np.full_like(p_map, np.nan)
    q_map[off_diag] = fdrcorrect_bh(p_map[off_diag])
    return q_map


def group_contrast(A, B):
    """Welch FT vs PT por arista y banda. A, B: (n_subj, n_bands, dest, src) -> mapas (dest, src, band)."""
    t_map, p_map = ttest_ind(A, B, axis=0, equal_var=False)
    t_map = np.transpose(np.nan_to_num(t_map), (1, 2, 0))
    p_map = np.transpose(p_map, (1, 2, 0))
    return t_map, p_map, fdr_over_edges(p_map)


def tfce_permutation_test(A, B, adjacency, args):
    """TFCE sobre |t| de Welch con permutación de etiquetas de grupo y estadístico máximo (FWER)."""
    t_abs = np.transpose(compute_welch_t_map(A, B), (1, 2, 0))
    tfce_real = tfce_transform(t_abs, adjacency, dh=args.dh)
    
    D_perm = np.concatenate([A, B], axis=0)
    timer = time.time()
    max_tfce_null = np.array(Parallel(n_jobs=args.jobs, backend="loky")(
        delayed(worker_permutation)(seed, D_perm, len(A), adjacency, args.dh)
        for seed in range(args.perms)
    ))
    print(f"  * {args.perms} permutaciones en {time.time() - timer:.2f} s")
    
    p_tfce = (1 + np.sum(max_tfce_null[None, None, None, :] >= tfce_real[..., None], axis=-1)) / (1 + args.perms)
    p_tfce[np.eye(A.shape[-1], dtype=bool)] = np.nan
    return p_tfce


def report(label, p_map, q_map):
    n_tests = int(np.sum(~np.isnan(q_map)))
    n_naive = int(np.nansum(np.where(np.isnan(q_map), np.nan, p_map) < 0.05))
    n_fdr = int(np.nansum(q_map < 0.05))
    print(f"  * {label}: {n_naive}/{n_tests} aristas con p < 0.05 (esperadas por azar: {0.05 * n_tests:.1f}) | {n_fdr} sobreviven FDR | q mínimo = {np.nanmin(q_map):.3f}")


def run(args):
    print("==========================================================================")
    print(" IPRE-2026: Pipeline de Análisis de Conectividad EEG")
    print(f" ROIs: {args.roi_method.upper()} | MVAR multi-ensayo a {parameters.FS_TARGET:.0f} Hz | Bandas: {', '.join(parameters.F_BANDS)}")
    print("==========================================================================")

    os.makedirs("plots", exist_ok=True)

    valid_subjects = get_valid_subjects()
    print(f"Sujetos válidos encontrados (ambas condiciones): {len(valid_subjects)}")

    # --- Opcional: Selección de Orden MVAR (AIC / BIC) ---
    if args.select_order:
        order_res = find_dataset_optimal_order(valid_subjects, parameters.ROIS, max_p=parameters.MAX_P, method=args.roi_method)
        if order_res is not None:
            plot_order_selection_curves(
                np.array(order_res['lags']),
                np.array(order_res['mean_aic']),
                np.array(order_res['mean_bic']),
                np.array(order_res['aic_votes']),
                np.array(order_res['bic_votes']),
                output_dir="plots",
                suffix=f"_{args.roi_method}",
                method_label=f"ROI: {args.roi_method.upper()}"
            )
            print(f"  -> Actualizando p={order_res['recommended_p']} para el cálculo de conectividad.")
            args.p = order_res['recommended_p']

    band_names = list(parameters.F_BANDS.keys())

    # --- 1. dDTF por sujeto y condición (un MVAR por registro con todas sus épocas) ---
    print(f"\n--- 1. Cálculo de Conectividad dDTF (p={args.p}) ---")
    HB_list, SI_list, groups, kid_ids, excluded = [], [], [], [], []
    for kid_id, info in valid_subjects.items():
        res_hb = load_and_compute_ddtf(info['hb'], p=args.p, roi_method=args.roi_method)
        res_si = load_and_compute_ddtf(info['si'], p=args.p, roi_method=args.roi_method)

        roots = (float(res_hb['max_root']), float(res_si['max_root']))
        if max(roots) >= 1.0:
            excluded.append((kid_id, roots))
            if not args.keep_unstable:
                continue

        HB_list.append(band_average(res_hb['ddtf']))
        SI_list.append(band_average(res_si['ddtf']))
        groups.append(info['group'])
        kid_ids.append(kid_id)

    print(f"  * Registros con MVAR inestable (raíz >= 1): {len(excluded)} sujetos {'(incluidos)' if args.keep_unstable else '(excluidos)'}")
    for kid_id, roots in excluded:
        print(f"      - {kid_id}: max_root hb={roots[0]:.4f}, si={roots[1]:.4f}")

    HB_all = np.array(HB_list)                   # (n_subj, n_bands, dest, src)
    SI_all = np.array(SI_list)
    D_all = HB_all - SI_all                      # contraste intra-sujeto: heartbeat - silencio
    M_all = (HB_all + SI_all) / 2                # nivel basal de conectividad (promedio de condiciones)
    groups = np.array(groups)
    D_FT_arr = D_all[groups == 'FT']
    D_PT_arr = D_all[groups == 'PT']
    print(f"  * Sujetos analizados: FT = {len(D_FT_arr)}, PT = {len(D_PT_arr)}")

    if len(D_FT_arr) < 2 or len(D_PT_arr) < 2:
        print("[ERROR] No hay datos suficientes para ambos grupos (FT y PT). Verifica las rutas.")
        return

    sfx = f"_{args.roi_method}"
    node_names = list(parameters.ROI_NAMES)
    np.savez(f"plots/ddtf_contrast{sfx}.npz", HB=HB_all, SI=SI_all, D=D_all, groups=groups, kid_ids=np.array(kid_ids),
             band_names=np.array(band_names), node_names=np.array(node_names))
    coords_3d, _ = get_roi_3d_centroids(parameters.ELP_FILE, parameters.ROIS)

    # --- 2. Chequeo de cordura: efecto de condición (hb vs si) en todos los sujetos ---
    print("\n--- 2. Efecto de Condición (Heartbeat vs Silencio, t pareado sobre todos los sujetos) ---")
    t_cond, p_cond = ttest_1samp(D_all, 0.0, axis=0)
    t_cond = np.transpose(np.nan_to_num(t_cond), (1, 2, 0))  # (dest, src, band)
    p_cond = np.transpose(p_cond, (1, 2, 0))
    q_cond = fdr_over_edges(p_cond)
    report("Condición", p_cond, q_cond)
    np.save(f"plots/t_values_condition{sfx}.npy", t_cond)
    np.save(f"plots/p_values_condition{sfx}_naive.npy", p_cond)
    np.save(f"plots/p_values_condition{sfx}_fdr.npy", q_cond)
    plot_contrast_heatmaps(t_cond, p_cond, node_names, band_names,
                           f"Heartbeat - Silencio, todos los sujetos (n={len(D_all)}) [ROI: {args.roi_method.upper()}]",
                           f"plots/contrast_condition{sfx}.png", q_map=q_cond)

    # --- 3. Efecto principal de grupo: FT vs PT sobre el promedio (hb + si) / 2 ---
    print("\n--- 3. Efecto Principal de Grupo (FT vs PT sobre el promedio de condiciones, Welch) ---")
    t_main, p_main, q_main = group_contrast(M_all[groups == 'FT'], M_all[groups == 'PT'])
    report("Grupo", p_main, q_main)
    np.save(f"plots/t_values_group{sfx}.npy", t_main)
    np.save(f"plots/p_values_group{sfx}_naive.npy", p_main)
    np.save(f"plots/p_values_group{sfx}_fdr.npy", q_main)
    plot_contrast_heatmaps(t_main, p_main, node_names, band_names,
                           f"FT vs PT, promedio de condiciones, FT={len(D_FT_arr)} PT={len(D_PT_arr)} [ROI: {args.roi_method.upper()}]",
                           f"plots/contrast_group{sfx}.png", q_map=q_main)
    for b_idx, b_name in enumerate(band_names):
        export_interactive_3d_network(coords_3d, p_main[:, :, b_idx], node_names,
                                      filename=f"plots/red_group_naive{sfx}_{b_name}.html",
                                      title=f"FT vs PT (promedio de condiciones), banda {b_name}: p < 0.05 sin corregir")
        
    # Prueba global: dDTF medio sobre las 56 aristas, una prueba por banda (Bonferroni sobre bandas)
    off_diag = ~np.eye(len(node_names), dtype=bool)
    global_hb = HB_all[:, :, off_diag].mean(axis=-1)  # (n_subj, n_bands)
    global_si = SI_all[:, :, off_diag].mean(axis=-1)
    global_mean = (global_hb + global_si) / 2
    t_glob, p_glob = ttest_ind(global_mean[groups == 'FT'], global_mean[groups == 'PT'], equal_var=False)
    for b_idx, b_name in enumerate(band_names):
        p_bonf = min(1.0, p_glob[b_idx] * len(band_names))
        print(f"  * Global {b_name}: dDTF medio FT = {global_mean[groups == 'FT', b_idx].mean():.4f}, "
              f"PT = {global_mean[groups == 'PT', b_idx].mean():.4f} | t = {t_glob[b_idx]:+.2f}, "
              f"p = {p_glob[b_idx]:.4f}, p_Bonf = {p_bonf:.4f}")
    plot_group_condition_means(global_hb, global_si, groups, band_names,
                               f"plots/group_condition_global{sfx}.png",
                               title=f"dDTF medio (56 aristas) por grupo y condición [ROI: {args.roi_method.upper()}]")
    
    # --- 4. Interacción Grupo x Condición: FT vs PT sobre (hb - si) ---
    print("\n--- 4. Interacción Grupo x Condición (FT vs PT sobre hb - si, Welch) ---")
    t_int, p_int, q_int = group_contrast(D_FT_arr, D_PT_arr)
    report("Interacción", p_int, q_int)
    np.save(f"plots/t_values_interaction{sfx}.npy", t_int)
    np.save(f"plots/p_values_interaction{sfx}_naive.npy", p_int)
    np.save(f"plots/p_values_interaction{sfx}_fdr.npy", q_int)
    plot_contrast_heatmaps(t_int, p_int, node_names, band_names,
                           f"FT vs PT sobre (hb - si), FT={len(D_FT_arr)} PT={len(D_PT_arr)} [ROI: {args.roi_method.upper()}]",
                           f"plots/contrast_interaction{sfx}.png", q_map=q_int)

    # --- 5. TFCE (Permutaciones de Monte Carlo, estadístico máximo => FWER) ---
    if args.tfce:
        print("\n--- 5. Corrección TFCE (Monte Carlo) ---")
        r_val = args.R if args.R is not None else parameters.R_ROI_DEFAULT
        print(f"  * Construyendo grafo espacial con R={r_val} cm")
        adj_spatial = get_spatial_adjacency_matrix(node_names, parameters.ELP_FILE, r_val, rois_dict=parameters.ROIS)
        adjacency = build_edge_band_graph(adj_spatial, n_bands=len(band_names))
        
        for label, data in (("group", M_all), ("interaction", D_all)):
            p_tfce = tfce_permutation_test(data[groups == 'FT'], data[groups == 'PT'], adjacency, args)
            np.save(f"plots/p_values_{label}{sfx}_tfce.npy", p_tfce)
            print(f"  * TFCE {label}: {int(np.nansum(p_tfce < 0.05))} aristas con p_FWER < 0.05 | p mínimo = {np.nanmin(p_tfce):.3f}")
        
    # --- 6. Robustez (si existen ambos análisis mean y pca) ---
    _, correlations = plot_robustness_comparison(band_names=band_names, output_dir="plots")
    if correlations:
        for b_name, r in correlations.items():
            print(f"  * Robustez Mean vs PCA, banda {b_name}: r = {r:.2f}")

    print("\n¡PIPELINE COMPLETADO EXITOSAMENTE!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Conectividad EEG")
    parser.add_argument("--p", type=int, default=parameters.P_OPTIMO, help="Orden MVAR (default: parameters.P_OPTIMO)")
    parser.add_argument("--roi-method", type=str, choices=['mean', 'pca'], default=parameters.ROI_EXTRACTION_METHOD, help="Método intra-ROI: 'mean' o 'pca'")
    parser.add_argument("--select-order", action="store_true", help="Buscar empíricamente el orden óptimo MVAR vía AIC/BIC antes de correr")
    parser.add_argument("--keep-unstable", action="store_true", help="No excluir sujetos cuyo MVAR resulte inestable")
    parser.add_argument("--tfce", action="store_true", help="Correr además TFCE con permutaciones (FWER)")
    parser.add_argument("--dh", type=float, default=0.1, help="Paso discreto dh para la integral TFCE")
    parser.add_argument("--R", type=float, default=None, help="Radio espacial (cm) para adyacencia TFCE (None = parameters.R_ROI_DEFAULT)")
    parser.add_argument("--perms", type=int, default=1000, help="Número de permutaciones Monte Carlo")
    parser.add_argument("--jobs", type=int, default=-1, help="Número de cores para paralelizacion (-1 = todos)")

    args = parser.parse_args()
    run(args)
