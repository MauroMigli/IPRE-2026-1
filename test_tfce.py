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
    t0 = time.time()
    ep = clean_epochs(mne.io.read_epochs_eeglab(filepath, verbose=False))
    sf = ep.info['sfreq']
    ddtf = process_dDTF_global(ep.get_data(copy=False), sampling_freq=sf, p=parameters.P_OPTIMO)
    print(f"    [OK] dDTF calculado en {time.time()-t0:.1f} s")
    return ddtf, ep.ch_names

if __name__ == "__main__":
    print("==========================================================================")
    print(" Evaluando TFCE 4D (1-sample test) con 3 Instancias (FT vs PT, HB vs SI)")
    print("==========================================================================")
    
    # 3 Instancias de la prueba (cada instancia es un emparejamiento aleatorio de FT y PT)
    # Ejemplo de identificadores reales en las carpetas
    instances = [
        {"FT": "003", "PT": "005"},
        {"FT": "004", "PT": "006"},
        {"FT": "009", "PT": "007"}
    ]
    
    D_list = []
    global_ch_names = None
    
    # Vamos a procesar cada instancia y almacenar las matrices parciales (liberando memoria)
    for k, inst in enumerate(instances):
        print(f"\n--- Instancia {k+1}/3: FT {inst['FT']} vs PT {inst['PT']} ---")
        
        # Rutas de los 4 archivos de esta instancia
        paths = {
            "FT_hb": f"data/epch_heartbeat/epch_FT_hb_obs_{inst['FT']}.set",
            "FT_si": f"data/epch_silence/epch_FT_si_obs_{inst['FT']}.set",
            "PT_hb": f"data/epch_heartbeat/epch_PT_hb_obs_{inst['PT']}.set",
            "PT_si": f"data/epch_silence/epch_PT_si_obs_{inst['PT']}.set"
        }
        
        # Calculamos dDTF para los 4 archivos
        ddtf_FT_hb, ch_names = load_and_compute_ddtf(paths["FT_hb"])
        if global_ch_names is None: global_ch_names = ch_names
        ddtf_FT_si, _ = load_and_compute_ddtf(paths["FT_si"])
        ddtf_PT_hb, _ = load_and_compute_ddtf(paths["PT_hb"])
        ddtf_PT_si, _ = load_and_compute_ddtf(paths["PT_si"])
        
        # Encontramos la cantidad mínima de épocas dentro de ESTA instancia
        # Shape original de dDTF: (epoch, freq, dest, src)
        min_ep = min(len(ddtf_FT_hb), len(ddtf_FT_si), len(ddtf_PT_hb), len(ddtf_PT_si))
        
        # D_k(i,j,f,t) = [FT_hb - FT_si] - [PT_hb - PT_si] truncando a las épocas válidas
        D_k = (ddtf_FT_hb[:min_ep] - ddtf_FT_si[:min_ep]) - (ddtf_PT_hb[:min_ep] - ddtf_PT_si[:min_ep])
        D_list.append(D_k)
        
        # Limpiar memoria
        del ddtf_FT_hb, ddtf_FT_si, ddtf_PT_hb, ddtf_PT_si
        gc.collect()
        
    print("\n--- Unificando dimensiones temporales ---")
    # Para poder hacer la media sobre el eje de muestras (axis=0), 
    # todas las instancias D_k deben tener el mismo número de épocas.
    global_min_epochs = min(len(D) for D in D_list)
    print(f"Trunando la dimensión temporal globalmente a {global_min_epochs} épocas.")
    
    # Array final D de tamaño: (N_samples, epoch, freq, dest, src)
    # N_samples será 3 en este caso.
    D_array = np.array([D[:global_min_epochs] for D in D_list])
    
    print("\n1. Calculando Estadístico T tetradimensional (T = media(D) / SE(D))...")
    # compute_t_map devuelve shape: (epoch, freq, dest, src)
    t_map_raw = compute_t_map(D_array)
    
    # Reformatear de (epoch, freq, dest, src) a (dest, src, freq, epoch) 
    # que es el formato natural (i, j, f, t) que requiere tfce_transform
    T_map_4d = np.transpose(t_map_raw, (2, 3, 1, 0))
    print(f"Shape final T_map_4d: {T_map_4d.shape}")
    print(f"Max T-stat encontrado: {np.max(T_map_4d):.3f}")
    
    print("\n2. Calculando TFCE (Caso A: Adyacencia Total)...")
    t0 = time.time()
    tfce_total = tfce_transform(T_map_4d, spatial_adjacency='total', dh=0.1)
    print(f" -> Supremo TFCE (Total): {np.max(tfce_total):.3f} (Tiempo: {time.time()-t0:.2f} s)")
    
    print("\n3. Calculando TFCE (Caso B: Adyacencia Nula)...")
    t0 = time.time()
    tfce_null = tfce_transform(T_map_4d, spatial_adjacency='null', dh=0.1)
    print(f" -> Supremo TFCE (Nula): {np.max(tfce_null):.3f} (Tiempo: {time.time()-t0:.2f} s)")
    
    # ====== VISUALIZACION ======
    print("\n4. Generando visualizaciones interactivas (HTML)...")
    # Para plotear la red de 65x65 canales, proyectamos las máximas activaciones de freq y epoch.
    # tfce shape es (dest, src, freq, epoch)
    tfce_total_2d = np.max(tfce_total, axis=(2, 3))
    tfce_null_2d = np.max(tfce_null, axis=(2, 3))
    
    pval_total = tfce_to_pseudo_p(tfce_total_2d)
    pval_null = tfce_to_pseudo_p(tfce_null_2d)
    
    coords_3d = html_plotter.get_3d_positions(parameters.ELP_FILE, global_ch_names)
    os.makedirs("plots", exist_ok=True)
    
    html_plotter.export_interactive_3d_network(
        coords_3d, pval_total, global_ch_names,
        filename="plots/red_tfce_total_real.html",
        dropped_channels=parameters.DROPPED_CHANNELS
    )
    
    html_plotter.export_interactive_3d_network(
        coords_3d, pval_null, global_ch_names,
        filename="plots/red_tfce_null_real.html",
        dropped_channels=parameters.DROPPED_CHANNELS
    )
    
    print("¡Proceso completado exitosamente!")
