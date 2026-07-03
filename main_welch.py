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

if __name__ == "__main__":
    print("==========================================================================")
    print(" IPRE-2026: Pipeline Básico - Test de Welch (Sin Corregir por FWER)")
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
    # Aquí no lanzamos hilos paralelos porque podemos reutilizar la caché ya generada
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
    
    print("\n--- Calculando T-Test de Welch (Sin Corrección Múltiple) ---")
    # scipy.stats.ttest_ind computa el estadístico T y el valor p para dos muestras independientes.
    # axis=0 calcula esto a lo largo de los sujetos para cada (ep, b, dest, src).
    # equal_var=False realiza el test de Welch (asume varianzas diferentes).
    t_stat, p_values_raw = ttest_ind(D_FT_arr, D_PT_arr, axis=0, equal_var=False, nan_policy='omit')
    
    # t_stat y p_values_raw ahora tienen forma (ep, b, dest, src)
    # Pero el plot_brain_interactive espera shape (dest, src, band, epoch).
    # Hacemos la transposición correspondiente:
    p_values_4d = np.transpose(p_values_raw, (2, 3, 1, 0)) # shape final: (dest, src, band, epoch)
    
    os.makedirs("plots", exist_ok=True)
    np.save("plots/channel_names.npy", np.array(global_ch_names, dtype=object))
    
    out_file = "plots/p_values_welch_naive.npy"
    np.save(out_file, p_values_4d)
    
    print(f"\nResultados guardados en {out_file}")
    print(f"P-valor mínimo detectado (ingenuo): {np.nanmin(p_values_4d):.6f}")
    print("\n¡PIPELINE COMPLETADO CON ÉXITO!")
