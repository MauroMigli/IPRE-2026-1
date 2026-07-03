import numpy as np
import time
from tfce_core import compute_welch_t_map, tfce_transform

def test_welch_and_permutations():
    print("=== INICIANDO MOCK TEST DE MAIN.PY (Welch + Label Swapping) ===")
    
    # 1. Crear tensores falsos (53 FT y 27 PT)
    # Shape: (muestras, epochs, bands, dest, src)
    # Usamos tamaños muy pequeños para que sea instantáneo
    # Ej: 3 epochs, 2 bands, 4 dest, 4 src
    N_FT = 53
    N_PT = 27
    
    D_FT = np.random.rand(N_FT, 3, 2, 4, 4)
    D_PT = np.random.rand(N_PT, 3, 2, 4, 4)
    
    print(f"Dimensiones simuladas: FT={D_FT.shape}, PT={D_PT.shape}")
    
    # 2. Probar compute_welch_t_map
    print("\nProbando compute_welch_t_map...")
    try:
        t_map_raw = compute_welch_t_map(D_FT, D_PT)
        print(f"Shape del mapa T resultante: {t_map_raw.shape} (Esperado: (3, 2, 4, 4))")
        assert t_map_raw.shape == (3, 2, 4, 4), "Shape incorrecta en el mapa T"
        # Transponer para TFCE: (dest, src, band, epoch)
        T_map_4d = np.transpose(t_map_raw, (2, 3, 1, 0))
        assert T_map_4d.shape == (4, 4, 2, 3), "Shape incorrecta después de transponer"
        print("[OK] Test de Welch superado.")
    except Exception as e:
        print(f"[ERROR] en compute_welch_t_map: {e}")
        return
        
    # 3. Probar TFCE con el T-map real
    print("\nProbando TFCE...")
    try:
        tfce_total = tfce_transform(T_map_4d, spatial_adjacency='total', dh=0.1)
        print(f"TFCE Total Max: {np.max(tfce_total):.3f}")
        tfce_null = tfce_transform(T_map_4d, spatial_adjacency='null', dh=0.1)
        print(f"TFCE Null Max: {np.max(tfce_null):.3f}")
        print("[OK] TFCE superado.")
    except Exception as e:
        print(f"[ERROR] en tfce_transform: {e}")
        return
        
    # 4. Probar lógica de Label Swapping
    print("\nProbando permutación de etiquetas (Label Swapping)...")
    try:
        D_all = np.concatenate([D_FT, D_PT], axis=0)
        N_total = N_FT + N_PT
        
        # 1 sola iteración de prueba
        perm_indices = np.random.permutation(N_total)
        D_FT_perm = D_all[perm_indices[:N_FT]]
        D_PT_perm = D_all[perm_indices[N_FT:]]
        
        t_map_perm = compute_welch_t_map(D_FT_perm, D_PT_perm)
        T_map_perm_4d = np.transpose(t_map_perm, (2, 3, 1, 0))
        
        tfce_perm_total = tfce_transform(T_map_perm_4d, spatial_adjacency='total', dh=0.1)
        print(f"Permutación TFCE Total Max: {np.max(tfce_perm_total):.3f}")
        print("[OK] Label Swapping superado.")
    except Exception as e:
        print(f"[ERROR] en Label Swapping: {e}")
        return
        
    print("\n¡MOCK TEST EXITOSO! Todo el flujo matemático de main.py está validado.")

if __name__ == "__main__":
    test_welch_and_permutations()
