import numpy as np
import time
import mne
import os
import gc
from pathlib import Path

from tfce_core import compute_welch_t_map, tfce_transform
from dDTF import process_dDTF_global
import parameters

def clean_epochs(epochs):
    chans_in_data = epochs.ch_names
    chans_to_drop = [ch for ch in parameters.DROPPED_CHANNELS if ch in chans_in_data]
    if chans_to_drop:
        epochs.drop_channels(chans_to_drop)
    return epochs

def load_and_compute_ddtf(filepath):
    # Directorio de caché
    cache_dir = "data/ddtf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    base_name = os.path.basename(filepath).replace(".set", "")
    cache_file = os.path.join(cache_dir, f"{base_name}_ddtf.npy")
    channels_file = os.path.join(cache_dir, f"{base_name}_channels.npy")
    
    if os.path.exists(cache_file) and os.path.exists(channels_file):
        ddtf = np.load(cache_file)
        ch_names = np.load(channels_file, allow_pickle=True).tolist()
        return ddtf, ch_names
        
    t0 = time.time()
    ep = clean_epochs(mne.io.read_epochs_eeglab(filepath, verbose=False))
    sf = ep.info['sfreq']
    ddtf = process_dDTF_global(ep.get_data(copy=False), sampling_freq=sf, p=parameters.P_OPTIMO)
    
    np.save(cache_file, ddtf)
    np.save(channels_file, np.array(ep.ch_names, dtype=object))
    return ddtf, ep.ch_names

if __name__ == "__main__":
    print("==========================================================================")
    print(" IPRE-2026: Análisis Definitivo FT vs PT (Muestra Completa)")
    print(" Test Welch 2-Muestras Independientes con Permutación Montecarlo (TFCE)")
    print("==========================================================================")
    
    # 1. Parsear metadatos de sujetos desde la lista de parámetros
    subjects = {}
    for f in parameters.HEARTBEAT + parameters.SILENCE:
        stem = Path(f).stem
        parts = stem.split('_')
        # Ejemplo: epch_FT_hb_obs_069
        if len(parts) >= 5:
            group = parts[1] # 'FT' o 'PT'
            cond = parts[2]  # 'hb' o 'si'
            kid_id = parts[4]
            
            if kid_id not in subjects:
                subjects[kid_id] = {'group': group, 'hb': None, 'si': None}
            
            if cond == 'hb': subjects[kid_id]['hb'] = f
            elif cond == 'si': subjects[kid_id]['si'] = f

    # Filtrar solo aquellos que tengan ambas condiciones
    valid_subjects = {k: v for k, v in subjects.items() if v['hb'] and v['si']}
    print(f"\nSujetos válidos encontrados: {len(valid_subjects)}")
    
    # 2. Precálculo paralelo con Joblib
    all_paths = []
    for info in valid_subjects.values():
        all_paths.extend([info['hb'], info['si']])
        
    print(f"\n--- Iniciando Pre-cálculo paralelo de dDTF ({len(all_paths)} archivos) ---")
    from joblib import Parallel, delayed
    t_paralelo = time.time()
    Parallel(n_jobs=-1)(delayed(load_and_compute_ddtf)(p) for p in set(all_paths))
    print(f"--- Pre-cálculo finalizado en {time.time() - t_paralelo:.2f} segundos ---")
    
    # 3. Cálculo de contrastes por sujeto: D_s = HB - SI
    D_FT = []
    D_PT = []
    global_ch_names = None
    
    bands = parameters.F_BANDS
    band_names = list(bands.keys())
    n_bands = len(band_names)
    fs_global = parameters.FS_GLOBAL
    
    print("\n--- Computando Contrastes HB-SI por sujeto y colapsando a bandas ---")
    for kid_id, info in valid_subjects.items():
        ddtf_hb, ch = load_and_compute_ddtf(info['hb'])
        if global_ch_names is None: global_ch_names = ch
        ddtf_si, _ = load_and_compute_ddtf(info['si'])
        
        min_ep = min(len(ddtf_hb), len(ddtf_si))
        D_s = ddtf_hb[:min_ep] - ddtf_si[:min_ep]
        
        n_ep, _, n_dest, n_src = D_s.shape
        D_s_bands = np.zeros((n_ep, n_bands, n_dest, n_src))
        
        for b_idx, b_name in enumerate(band_names):
            f_min, f_max = bands[b_name]
            freq_mask = (fs_global >= f_min) & (fs_global <= f_max)
            if np.any(freq_mask):
                D_s_bands[:, b_idx, :, :] = np.mean(D_s[:, freq_mask, :, :], axis=1)
                
        if info['group'] == 'FT': D_FT.append(D_s_bands)
        elif info['group'] == 'PT': D_PT.append(D_s_bands)
        
        del ddtf_hb, ddtf_si, D_s, D_s_bands
        
    gc.collect()
    
    # Unificar dimensión temporal (épocas) globalmente
    print("\n--- Unificando dimensiones temporales ---")
    global_min_epochs = min([len(D) for D in D_FT] + [len(D) for D in D_PT])
    print(f"Truncando a un mínimo global de {global_min_epochs} épocas.")
    
    D_FT_arr = np.array([D[:global_min_epochs] for D in D_FT]) # (53, epoch, band, dest, src)
    D_PT_arr = np.array([D[:global_min_epochs] for D in D_PT]) # (27, epoch, band, dest, src)
    
    print(f"Muestras FT: {len(D_FT_arr)}")
    print(f"Muestras PT: {len(D_PT_arr)}")
    
    # 4. Cálculo Estadístico Real
    print("\n1. Calculando Test T (Welch) y TFCE para los datos originales...")
    t_map_raw = compute_welch_t_map(D_FT_arr, D_PT_arr) # Absoluto para 2-colasy
    T_map_4d = np.transpose(t_map_raw, (2, 3, 1, 0)) # (dest, src, band, epoch)
    
    t0 = time.time()
    tfce_real_total = tfce_transform(T_map_4d, spatial_adjacency='total', dh=0.1)
    tfce_real_null = tfce_transform(T_map_4d, spatial_adjacency='null', dh=0.1)
    print(f" -> TFCE original completado en {time.time()-t0:.2f} s")
    
    # 5. Permutaciones Montecarlo (Label Swapping)
    K_perms = 1000
    N_FT = len(D_FT_arr)
    N_PT = len(D_PT_arr)
    N_total = N_FT + N_PT
    
    D_all = np.concatenate([D_FT_arr, D_PT_arr], axis=0)
    
    print(f"\n2. Ejecutando Permutaciones Montecarlo ({K_perms} iteraciones, Label Swapping)...")
    
    supremos_total = []
    supremos_null = []
    
    t_perm_start = time.time()
    for k in range(K_perms):
        # Permutar etiquetas
        perm_indices = np.random.permutation(N_total)
        D_FT_perm = D_all[perm_indices[:N_FT]]
        D_PT_perm = D_all[perm_indices[N_FT:]]
        
        # Test de Welch permutado
        t_map_perm = compute_welch_t_map(D_FT_perm, D_PT_perm)
        T_map_perm_4d = np.transpose(t_map_perm, (2, 3, 1, 0))
        
        tfce_perm_total = tfce_transform(T_map_perm_4d, spatial_adjacency='total', dh=0.1)
        tfce_perm_null = tfce_transform(T_map_perm_4d, spatial_adjacency='null', dh=0.1)
        
        supremos_total.append(np.max(tfce_perm_total))
        supremos_null.append(np.max(tfce_perm_null))
        
        if (k + 1) % 10 == 0 or k == K_perms - 1:
            elapsed = time.time() - t_perm_start
            print(f"  -> Iteración {k+1}/{K_perms} procesada ({elapsed:.1f} s). Max Total: {supremos_total[-1]:.3f} | Max Nulo: {supremos_null[-1]:.3f}")
            
    # 6. Cálculo de p-valores FWER
    print("\n3. Calculando P-valores empíricos FWER...")
    supremos_total = np.array(supremos_total)
    supremos_null = np.array(supremos_null)
    
    p_values_total = (np.sum(supremos_total[:, None, None, None, None] >= tfce_real_total[None, ...], axis=0) + 1) / (K_perms + 1)
    p_values_null = (np.sum(supremos_null[:, None, None, None, None] >= tfce_real_null[None, ...], axis=0) + 1) / (K_perms + 1)
    
    # 7. Exportación
    print("\n4. Generando archivos de salida...")
    os.makedirs("plots", exist_ok=True)
    np.save("plots/p_values_empiricos_total.npy", p_values_total)
    np.save("plots/p_values_empiricos_null.npy", p_values_null)
    np.save("plots/channel_names.npy", np.array(global_ch_names, dtype=object))
    print(" -> Matrices .npy guardadas en plots/")
    print("¡Proceso principal finalizado con éxito!")