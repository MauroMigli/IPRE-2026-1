import numpy as np
import time
import mne
import os
import gc

from tfce_core import compute_t_map, tfce_transform
from dDTF import process_dDTF_global
import parameters
import html_plotter

def clean_epochs(epochs):
    chans_in_data = epochs.ch_names
    chans_to_drop = [ch for ch in parameters.DROPPED_CHANNELS if ch in chans_in_data]
    if chans_to_drop:
        epochs.drop_channels(chans_to_drop)
    return epochs

def tfce_to_pseudo_p(tfce_2d):
    """
    Convierte valores TFCE positivos en pseudo p-valores 
    para compatibilidad con html_plotter (donde p<0.05 es significativo).
    """
    p_vals = np.ones_like(tfce_2d)
    max_t = np.max(tfce_2d)
    if max_t > 0:
        norm_tfce = tfce_2d / max_t
        p_vals = np.exp(-norm_tfce * 10)  
    return p_vals

def load_and_compute_ddtf(filepath):
    print(f" -> Procesando {os.path.basename(filepath)}...")
    
    # Crear directorio de caché si no existe
    cache_dir = "data/ddtf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Definir rutas de caché para dDTF y canales
    base_name = os.path.basename(filepath).replace(".set", "")
    cache_file = os.path.join(cache_dir, f"{base_name}_ddtf.npy")
    channels_file = os.path.join(cache_dir, f"{base_name}_channels.npy")
    
    if os.path.exists(cache_file) and os.path.exists(channels_file):
        print(f"    [CACHÉ] Cargando dDTF desde caché para {base_name}...")
        t0 = time.time()
        ddtf = np.load(cache_file)
        ch_names = np.load(channels_file, allow_pickle=True).tolist()
        print(f"    [OK] dDTF cargado en {time.time()-t0:.2f} s")
        return ddtf, ch_names
        
    t0 = time.time()
    ep = clean_epochs(mne.io.read_epochs_eeglab(filepath, verbose=False))
    sf = ep.info['sfreq']
    ddtf = process_dDTF_global(ep.get_data(copy=False), sampling_freq=sf, p=parameters.P_OPTIMO)
    
    # Guardar en caché
    np.save(cache_file, ddtf)
    np.save(channels_file, np.array(ep.ch_names, dtype=object))
    
    print(f"    [OK] dDTF calculado y guardado en caché en {time.time()-t0:.1f} s")
    return ddtf, ep.ch_names

if __name__ == "__main__":
    import itertools
    print("==========================================================================")
    print(" Evaluando TFCE 4D Colapsado a Bandas con Montecarlo (Sign Flip)")
    print("==========================================================================")
    
    # 6 Instancias de la prueba (cada instancia es un emparejamiento aleatorio de FT y PT)
    instances = [
        {"FT": "003", "PT": "005"},
        {"FT": "004", "PT": "006"},
        {"FT": "009", "PT": "007"},
        {"FT": "013", "PT": "008"},
        {"FT": "014", "PT": "010"},
        {"FT": "016", "PT": "011"}
    ]
    
    D_list = []
    global_ch_names = None
    
    # Bandas de frecuencia predefinidas
    fs_global = parameters.FS_GLOBAL
    bands = parameters.F_BANDS
    band_names = list(bands.keys())
    n_bands = len(band_names)
    
    print("\n--- Pre-calculando dDTF en paralelo (Caché) ---")
    all_paths = []
    for inst in instances:
        all_paths.extend([
            f"data/epch_heartbeat/epch_FT_hb_obs_{inst['FT']}.set",
            f"data/epch_silence/epch_FT_si_obs_{inst['FT']}.set",
            f"data/epch_heartbeat/epch_PT_hb_obs_{inst['PT']}.set",
            f"data/epch_silence/epch_PT_si_obs_{inst['PT']}.set"
        ])
    
    from joblib import Parallel, delayed
    # Usamos n_jobs=-1 para ocupar todas las CPUs que SLURM haya asignado
    Parallel(n_jobs=-1)(delayed(load_and_compute_ddtf)(p) for p in set(all_paths))
    print("--- Pre-cálculo finalizado ---\n")
    
    for k, inst in enumerate(instances):
        print(f"\n--- Instancia {k+1}/{len(instances)}: FT {inst['FT']} vs PT {inst['PT']} ---")
        
        paths = {
            "FT_hb": f"data/epch_heartbeat/epch_FT_hb_obs_{inst['FT']}.set",
            "FT_si": f"data/epch_silence/epch_FT_si_obs_{inst['FT']}.set",
            "PT_hb": f"data/epch_heartbeat/epch_PT_hb_obs_{inst['PT']}.set",
            "PT_si": f"data/epch_silence/epch_PT_si_obs_{inst['PT']}.set"
        }
        
        ddtf_FT_hb, ch_names = load_and_compute_ddtf(paths["FT_hb"])
        if global_ch_names is None: global_ch_names = ch_names
        ddtf_FT_si, _ = load_and_compute_ddtf(paths["FT_si"])
        ddtf_PT_hb, _ = load_and_compute_ddtf(paths["PT_hb"])
        ddtf_PT_si, _ = load_and_compute_ddtf(paths["PT_si"])
        
        min_ep = min(len(ddtf_FT_hb), len(ddtf_FT_si), len(ddtf_PT_hb), len(ddtf_PT_si))
        
        # Diferencia de conectividad: (FT_hb - FT_si) - (PT_hb - PT_si)
        # shape: (epoch, freq, dest, src)
        D_k = (ddtf_FT_hb[:min_ep] - ddtf_FT_si[:min_ep]) - (ddtf_PT_hb[:min_ep] - ddtf_PT_si[:min_ep])
        
        # Colapsar eje de frecuencia a bandas
        n_ep, _, n_dest, n_src = D_k.shape
        D_k_bands = np.zeros((n_ep, n_bands, n_dest, n_src))
        
        for b_idx, b_name in enumerate(band_names):
            f_min, f_max = bands[b_name]
            freq_mask = (fs_global >= f_min) & (fs_global <= f_max)
            if np.any(freq_mask):
                D_k_bands[:, b_idx, :, :] = np.mean(D_k[:, freq_mask, :, :], axis=1)
                
        D_list.append(D_k_bands)
        
        del ddtf_FT_hb, ddtf_FT_si, ddtf_PT_hb, ddtf_PT_si, D_k
        gc.collect()
        
    print("\n--- Unificando dimensiones temporales ---")
    global_min_epochs = min(len(D) for D in D_list)
    print(f"Trunando la dimensión temporal globalmente a {global_min_epochs} épocas.")
    
    # Array final D de tamaño: (N_samples, epoch, band, dest, src)
    D_array = np.array([D[:global_min_epochs] for D in D_list])
    
    print("\n1. Calculando Estadísticos T y TFCE empíricos originales...")
    t_map_raw = compute_t_map(D_array) # shape: (epoch, band, dest, src)
    T_map_4d = np.transpose(t_map_raw, (2, 3, 1, 0)) # a (dest, src, band, epoch)
    
    t0 = time.time()
    tfce_real_total = tfce_transform(T_map_4d, spatial_adjacency='total', dh=0.1)
    print(f" -> TFCE original (Total) completado en {time.time()-t0:.2f} s. (Supremo: {np.max(tfce_real_total):.3f})")
    
    t0 = time.time()
    tfce_real_null = tfce_transform(T_map_4d, spatial_adjacency='null', dh=0.1)
    print(f" -> TFCE original (Nula) completado en {time.time()-t0:.2f} s. (Supremo: {np.max(tfce_real_null):.3f})")
    
    print("\n2. Ejecutando Permutaciones Montecarlo (Sign Flips) para Total y Nula...")
    N = len(instances)
    K_perms = 1000  # Número de iteraciones Montecarlo
    if 2**N <= K_perms:
        # Exhaustivo si el N es muy pequeño
        sign_flips = list(itertools.product([-1, 1], repeat=N))
        N_perms = len(sign_flips)
        print(f"Universo manejable (2^{N} <= {K_perms}). Realizando permutación exhaustiva: {N_perms} iteraciones.")
    else:
        # Muestreo aleatorio
        print(f"Universo grande (2^{N} > {K_perms}). Realizando muestreo Montecarlo aleatorio con {K_perms} iteraciones.")
        sign_flips = np.random.choice([-1, 1], size=(K_perms, N))
        N_perms = K_perms
    
    supremos_nulos_total = []
    supremos_nulos_null = []
    
    t_perm_start = time.time()
    for p_idx, signs in enumerate(sign_flips):
        signs_arr = np.array(signs)[:, None, None, None, None]
        D_perm = D_array * signs_arr
        
        t_map_perm = compute_t_map(D_perm)
        T_map_perm_4d = np.transpose(t_map_perm, (2, 3, 1, 0))
        
        # Calcular TFCE para ambos casos
        tfce_perm_total = tfce_transform(T_map_perm_4d, spatial_adjacency='total', dh=0.1)
        tfce_perm_null = tfce_transform(T_map_perm_4d, spatial_adjacency='null', dh=0.1)
        
        supremos_nulos_total.append(np.max(tfce_perm_total))
        supremos_nulos_null.append(np.max(tfce_perm_null))
        
        if (p_idx + 1) % 10 == 0 or p_idx == N_perms - 1:
            elapsed = time.time() - t_perm_start
            print(f"  -> Permutación {p_idx+1}/{N_perms} procesada ({elapsed:.1f} s). Max Total: {np.max(tfce_perm_total):.3f} | Max Nulo: {np.max(tfce_perm_null):.3f}")
            
    supremos_nulos_total = np.array(supremos_nulos_total)
    supremos_nulos_null = np.array(supremos_nulos_null)
    
    print("\n3. Calculando P-valores empíricos FWER (Ajuste Davison-Hinkley)...")
    p_values_total = (np.sum(supremos_nulos_total[:, None, None, None, None] >= tfce_real_total[None, ...], axis=0) + 1) / (N_perms + 1)
    p_values_null = (np.sum(supremos_nulos_null[:, None, None, None, None] >= tfce_real_null[None, ...], axis=0) + 1) / (N_perms + 1)
    
    # ====== VISUALIZACION ======
    print("\n4. Generando visualizaciones...")
    os.makedirs("plots", exist_ok=True)
    
    # Guardar las matrices 4D completas y los canales para graficado interactivo posterior
    np.save("plots/p_values_empiricos_total.npy", p_values_total)
    np.save("plots/p_values_empiricos_null.npy", p_values_null)
    np.save("plots/channel_names.npy", np.array(global_ch_names, dtype=object))
    print(" -> Matrices 4D de p-valores y nombres de canales guardados en plots/ (.npy)")
    
    print("¡Proceso completado exitosamente!")
