import numpy as np
import os
import time
import sys

# Dynamically add the current directory to path
repo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_dir)

import parameters
from tfce_core import tfce_transform, get_spatial_adjacency_matrix, build_4d_graph
import save_t_map

def main():
    os.chdir(repo_dir)
    t_map_path = "plots/T_map_real.npy"
    
    if not os.path.exists(t_map_path):
        print("El T-Map real no existe. Extrayéndolo de los datos crudos (puede tomar ~1-2 min)...")
        save_t_map.main()
        
    print("Cargando T-Map Real...")
    T_map_real_4d = np.load(t_map_path)
    n_dest, n_src, n_bands, n_epochs = T_map_real_4d.shape
    
    ch_names = np.load("plots/channel_names.npy", allow_pickle=True)
    
    # Definir los radios solicitados
    radios = {
        0.00: 'null',
        6.44: 'build',
        19.32: 'total'
    }
    
    os.makedirs("plots/tfce_raw", exist_ok=True)
    
    for R, mode in radios.items():
        print(f"\n--- Procesando TFCE para R = {R} ---")
        t0 = time.time()
        
        if mode == 'null':
            adj_4d = 'null'
        elif mode == 'total':
            adj_4d = 'total'
        else:
            ch_adj = get_spatial_adjacency_matrix(ch_names, parameters.ELP_FILE, R)
            adj_4d = build_4d_graph(ch_adj, n_bands, n_epochs)
            
        tfce_real = tfce_transform(T_map_real_4d, spatial_adjacency=adj_4d, dh=0.1)
        
        # Aplicamos la transformacion logaritmica (log1p)
        tfce_log = np.log1p(tfce_real)
        
        # Guardamos el archivo
        out_file = f"plots/tfce_raw/tfce_log_R_{R:.2f}.npy"
        np.save(out_file, tfce_log)
        
        print(f"TFCE (log1p) para R={R:.2f} calculated en {time.time()-t0:.2f}s")
        print(f"Guardado en {out_file}")

if __name__ == '__main__':
    main()
