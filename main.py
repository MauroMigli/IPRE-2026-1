import numpy as np
import time
import mne
import os
import gc
from pathlib import Path
from joblib import Parallel, delayed

from tfce_core import compute_welch_t_map, tfce_transform, get_spatial_adjacency_matrix, build_4d_graph
from dDTF import process_dDTF_global
from html_plotter import get_3d_positions
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

def worker_permutation(seed, D_all, N_FT, N_total, adj_4d, dh):
    """Worker para procesar una permutación Montecarlo en paralelo."""
    rng = np.random.default_rng(seed) # Thread-safe Generator
    perm_indices = rng.permutation(N_total)
    D_FT_perm = D_all[perm_indices[:N_FT]]
    D_PT_perm = D_all[perm_indices[N_FT:]]
    
    t_map_perm = compute_welch_t_map(D_FT_perm, D_PT_perm)
    T_map_perm_4d = np.transpose(t_map_perm, (2, 3, 1, 0))
    
    tfce_perm = tfce_transform(T_map_perm_4d, spatial_adjacency=adj_4d, dh=dh)
    return np.max(tfce_perm)

if __name__ == "__main__":
    print("==========================================================================")
    print(" IPRE-2026: Pipeline TFCE Definitivo - Búsqueda de Radio R (Paralelo)")
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
    
    # 2. Precálculo paralelo dDTF
    all_paths = []
    for info in valid_subjects.values():
        all_paths.extend([info['hb'], info['si']])
        
    def cache_ddtf(filepath):
        load_and_compute_ddtf(filepath)
        return None

    print(f"\n--- Verificando Caché dDTF ({len(all_paths)} archivos) ---")
    # Limitamos n_jobs=1 en esta fase porque process_dDTF_global consume muchísima RAM
    # Usamos cache_ddtf para retornar None y evitar que joblib acumule 18 GB de RAM
    Parallel(n_jobs=1)(delayed(cache_ddtf)(p) for p in set(all_paths))
    
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
    
    n_bands = len(band_names)
    
    # Preparación Matemática Base (Test de Welch T-map real)
    t_map_raw = compute_welch_t_map(D_FT_arr, D_PT_arr)
    T_map_real_4d = np.transpose(t_map_raw, (2, 3, 1, 0)) # (dest, src, band, epoch)
    
    D_all = np.concatenate([D_FT_arr, D_PT_arr], axis=0)
    N_FT = len(D_FT_arr)
    N_total = len(D_all)
    K_perms = 1000
    
    # 4. Cálculo de Grilla de Radios (R)
    print("\n--- Calculando Grilla de Radios (R) ---")
    coords = get_3d_positions(parameters.ELP_FILE, global_ch_names)
    
    # Encontrar distancia máxima entre electrodos (D_max)
    max_dist = 0
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            d = np.linalg.norm(coords[i] - coords[j])
            if d > max_dist: max_dist = d
            
    # 10 valores de R desde 0 (Null) hasta max_dist (Total)
    R_values = np.linspace(0, max_dist, 10)
    print(f"Distancia máxima detectada: {max_dist:.2f}")
    print(f"Radios a evaluar: {['%.2f'%r for r in R_values]}")
    
    os.makedirs("plots", exist_ok=True)
    np.save("plots/channel_names.npy", np.array(global_ch_names, dtype=object))
    
    # BUCLE EXTERNO DE BÚSQUEDA R
    for idx, R in enumerate(R_values):
        print(f"\n==================================================")
        print(f" Iniciando procesamiento para R={R:.2f} (Paso {idx+1}/10)")
        print(f"==================================================")
        
        # A) Construir grafo disperso o usar atajos optimizados
        t0 = time.time()
        ch_adj = get_spatial_adjacency_matrix(global_ch_names, parameters.ELP_FILE, R)
        
        # Atajo 1: Si R es 0, es equivalente a la adyacencia nula
        if R == 0:
            adj_4d = 'null'
            print("Radio R=0 detectado. Usando optimización 'null' (0 bytes en RAM).")
        # Atajo 2: Si todos los canales están conectados, es equivalente a la adyacencia total
        elif np.all(ch_adj):
            adj_4d = 'total'
            print("El radio R conecta todos los canales de la red. Usando optimización 'total' (0 bytes en RAM).")
        else:
            adj_4d = build_4d_graph(ch_adj, n_bands, global_min_epochs)
            print(f"Grafo de {adj_4d.nnz} conexiones construido en {time.time()-t0:.2f}s")
        
        # B) TFCE Real
        t0 = time.time()
        tfce_real = tfce_transform(T_map_real_4d, spatial_adjacency=adj_4d, dh=0.1)
        print(f"TFCE Real calculado en {time.time()-t0:.2f}s (Max: {np.max(tfce_real):.3f})")
        
        # C) Montecarlo Paralelo usando Hilos (backend='threading') para compartir memoria nativamente y evitar OOM
        # Leemos los cores asignados por SLURM (por defecto 8) y limitamos a un máximo seguro de 4 para no saturar la RAM de 16G
        slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
        n_workers = min(4, slurm_cpus)
        
        print(f"Ejecutando {K_perms} permutaciones en paralelo ({n_workers} hilos)...")
        t0 = time.time()
        
        seeds = np.random.randint(0, 1000000, size=K_perms)
        
        supremos = Parallel(n_jobs=n_workers, backend='threading')(
            delayed(worker_permutation)(s, D_all, N_FT, N_total, adj_4d, 0.1)
            for s in seeds
        )
        print(f"Montecarlo completado en {time.time()-t0:.2f}s")
        
        # D) P-valores
        supremos = np.array(supremos)
        p_values = (np.sum(supremos[:, None, None, None, None] >= tfce_real[None, ...], axis=0) + 1) / (K_perms + 1)
        
        # Guardado
        out_file = f"plots/p_values_R_{idx}_val_{R:.2f}.npy"
        np.save(out_file, p_values)
        print(f"Resultados guardados en {out_file}")
        
        # E) Limpiar Memoria
        del adj_4d, ch_adj, supremos, p_values, tfce_real
        gc.collect()

    print("\n¡BÚSQUEDA INTENSIVA COMPLETADA CON ÉXITO!")