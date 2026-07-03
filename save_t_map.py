import numpy as np
from pathlib import Path
import os
import gc
import parameters
from tfce_core import compute_welch_t_map
from main import load_and_compute_ddtf

def main():
    print("=== Extrayendo Mapa T de Welch Real ===")
    
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
    
    D_FT, D_PT = [], []
    global_ch_names = None
    bands = parameters.F_BANDS
    band_names = list(bands.keys())
    fs_global = parameters.FS_GLOBAL
    
    # 2. Cargar datos del caché
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
    
    # 3. Unificar épocas y calcular Welch
    global_min_epochs = min([len(D) for D in D_FT] + [len(D) for D in D_PT])
    D_FT_arr = np.array([D[:global_min_epochs] for D in D_FT])
    D_PT_arr = np.array([D[:global_min_epochs] for D in D_PT])
    
    print("Calculando T-map de Welch...")
    t_map_raw = compute_welch_t_map(D_FT_arr, D_PT_arr)
    T_map_real_4d = np.transpose(t_map_raw, (2, 3, 1, 0)) # (dest, src, band, epoch)
    
    os.makedirs("plots", exist_ok=True)
    out_path = "plots/T_map_real.npy"
    np.save(out_path, T_map_real_4d)
    print(f"¡Mapa T guardado exitosamente en '{out_path}'!")

if __name__ == "__main__":
    main()
