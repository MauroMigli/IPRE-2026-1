import numpy as np
import time
import mne
import os
import gc
from pathlib import Path
from scipy.stats import ttest_ind

from dDTF import process_dDTF_global
import parameters

def clean_epochs(epochs):
    chans_in_data = epochs.ch_names
    chans_to_drop = [ch for ch in parameters.DROPPED_CHANNELS if ch in chans_in_data]
    if chans_to_drop:
        epochs.drop_channels(chans_to_drop)
    return epochs

def load_and_compute_ddtf(filepath):
    cache_dir = "data/ddtf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    base_name = os.path.basename(filepath).replace(".set", "")
    cache_file = os.path.join(cache_dir, f"{base_name}_ddtf.npy")
    channels_file = os.path.join(cache_dir, f"{base_name}_channels.npy")
    
    if os.path.exists(cache_file) and os.path.exists(channels_file):
        ddtf = np.load(cache_file)
        ch_names = np.load(channels_file, allow_pickle=True).tolist()
        return ddtf, ch_names
        
    ep = clean_epochs(mne.io.read_epochs_eeglab(filepath, verbose=False))
    sf = ep.info['sfreq']
    ddtf = process_dDTF_global(ep.get_data(copy=False), sampling_freq=sf, p=parameters.P_OPTIMO)
    
    np.save(cache_file, ddtf)
    np.save(channels_file, np.array(ep.ch_names, dtype=object))
    return ddtf, ep.ch_names

def fdrcorrect_bh(pvals):
    """
    Corrección de False Discovery Rate (FDR) usando el procedimiento de Benjamini-Hochberg.
    Devuelve los p-valores ajustados (q-valores).
    """
    pvals = np.asanyarray(pvals)
    n = len(pvals)
    if n == 0:
        return pvals
        
    sort_idx = np.argsort(pvals)
    pvals_sorted = pvals[sort_idx]
    
    adj_pvals = np.zeros(n)
    prev_q = 1.0
    for i in range(n - 1, -1, -1):
        q = pvals_sorted[i] * n / (i + 1)
        q = min(q, prev_q)
        adj_pvals[i] = q
        prev_q = q
        
    adj_pvals_orig = np.zeros(n)
    adj_pvals_orig[sort_idx] = adj_pvals
    return np.clip(adj_pvals_orig, 0.0, 1.0)

if __name__ == "__main__":
    print("==========================================================================")
    print(" IPRE-2026: Pipeline Comparativo - Welch (Naive) & Welch + FDR (Global)")
    print("==========================================================================")
    
    # 1. Parsear metadatos
    subjects = {}
    for f in parameters.HEARTBEAT + parameters.SILENCE:
        stem = Path(f).stem
        parts = stem.split('_')
        if len(parts) >= 5:
            group, cond, kid_id = parts[1], parts[2], parts[4]
            if kid_id not in subjects:
                subjects[kid_id] = {'group': group, 'hb': None, 'si': None}
            if cond == 'hb': subjects[kid_id]['hb'] = f
            elif cond == 'si': subjects[kid_id]['si'] = f

    valid_subjects = {k: v for k, v in subjects.items() if v['hb'] and v['si']}
    
    # 2. Precálculo dDTF
    all_paths = []
    for info in valid_subjects.values():
        all_paths.extend([info['hb'], info['si']])
        
    print(f"\n--- Verificando Caché dDTF ({len(all_paths)} archivos) ---")
    for p in set(all_paths):
        load_and_compute_ddtf(p)
    
    # 3. Cálculo de contrastes D_s = HB - SI
    D_FT, D_PT = [], []
    global_ch_names = None
    
    bands = parameters.F_BANDS
    band_names = list(bands.keys())
    fs_global = parameters.FS_GLOBAL
    
    print("\n--- Computando Contrastes HB-SI por sujeto ---")
    for kid_id, info in valid_subjects.items():
        ddtf_hb, ch = load_and_compute_ddtf(info['hb'])
        if global_ch_names is None: global_ch_names = ch
        ddtf_si, _ = load_and_compute_ddtf(info['si'])
        
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
    
    # Unificar temporalidad
    global_min_epochs = min([len(D) for D in D_FT] + [len(D) for D in D_PT])
    D_FT_arr = np.array([D[:global_min_epochs] for D in D_FT]) # (53, ep, b, dest, src)
    D_PT_arr = np.array([D[:global_min_epochs] for D in D_PT]) # (27, ep, b, dest, src)
    
    print("\n--- Calculando T-Test de Welch Base ---")
    t_stat, p_values_raw = ttest_ind(D_FT_arr, D_PT_arr, axis=0, equal_var=False, nan_policy='omit')
    
    # Transponer a formato de salida: (dest, src, band, epoch)
    p_values_real_4d = np.transpose(p_values_raw, (2, 3, 1, 0))
    n_dest, n_src, n_bands, n_epochs = p_values_real_4d.shape
    
    # 4. Enmascarar la diagonal (auto-conexiones) con NaN
    for i in range(n_dest):
        p_values_real_4d[i, i, :, :] = np.nan
        
    # 5. Welch + FDR (Benjamini-Hochberg) global
    # Seleccionamos todos los elementos que no sean NaN para corregirlos
    valid_indices = np.where(~np.isnan(p_values_real_4d))
    pvals_to_correct = p_values_real_4d[valid_indices]
    
    print(f"Aplicando FDR (Benjamini-Hochberg) sobre {len(pvals_to_correct)} comparaciones...")
    corrected_pvals = fdrcorrect_bh(pvals_to_correct)
    
    # Reconstruir el tensor 4D para FDR
    p_values_fdr_4d = np.full_like(p_values_real_4d, np.nan)
    p_values_fdr_4d[valid_indices] = corrected_pvals
    
    # 6. Guardar archivos globales
    os.makedirs("plots", exist_ok=True)
    np.save("plots/channel_names.npy", np.array(global_ch_names, dtype=object))
    
    out_naive = "plots/p_values_welch_naive.npy"
    out_fdr = "plots/p_values_welch_fdr.npy"
    
    np.save(out_naive, p_values_real_4d)
    np.save(out_fdr, p_values_fdr_4d)
    
    print(f"\nResultados guardados exitosamente:")
    print(f"  -> Welch Naive: {out_naive} (Min p: {np.nanmin(p_values_real_4d):.6f})")
    print(f"  -> Welch + FDR: {out_fdr} (Min p: {np.nanmin(p_values_fdr_4d):.6f})")
    
    print("\n¡PROCESAMIENTO COMPARATIVO COMPLETADO CON ÉXITO!")
