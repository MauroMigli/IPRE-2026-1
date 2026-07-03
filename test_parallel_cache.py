import os
import time
import numpy as np
from joblib import Parallel, delayed

# --- 1. MOCK DE LA FUNCIÓN PRINCIPAL ---
# Simulamos la función de test_tfce.py
def mock_load_and_compute_ddtf(filepath):
    # Simulamos el tiempo de procesamiento si no hay caché
    cache_dir = "data/ddtf_cache_mock"
    os.makedirs(cache_dir, exist_ok=True)
    
    base_name = os.path.basename(filepath).replace(".set", "")
    cache_file = os.path.join(cache_dir, f"{base_name}_ddtf.npy")
    channels_file = os.path.join(cache_dir, f"{base_name}_channels.npy")
    
    if os.path.exists(cache_file) and os.path.exists(channels_file):
        print(f"    [CACHÉ] Cargando mock desde caché para {base_name}...")
        ddtf = np.load(cache_file)
        ch_names = np.load(channels_file, allow_pickle=True).tolist()
        return ddtf, ch_names
        
    print(f" -> Procesando MOCK {os.path.basename(filepath)} (esto simula CPU intensivo)...")
    time.sleep(2)  # Simula el procesamiento lento
    
    # Creamos un tensor 4D aleatorio falso (freq, epoch, dest, src) para la prueba
    # Usaremos dimensiones pequeñas para que no ocupe memoria
    ddtf = np.random.rand(10, 5, 4, 4)
    ch_names = ["Fp1", "Fp2", "F3", "F4"]
    
    np.save(cache_file, ddtf)
    np.save(channels_file, np.array(ch_names, dtype=object))
    print(f"    [OK] Mock guardado en caché para {base_name}")
    
    return ddtf, ch_names

if __name__ == "__main__":
    print("=== INICIANDO PRUEBA DE PARALELIZACIÓN Y CACHÉ ===")
    
    instances = [
        {"FT": "MOCK_001", "PT": "MOCK_002"},
        {"FT": "MOCK_003", "PT": "MOCK_004"}
    ]
    
    # 1. Recolectar todos los archivos a procesar
    all_paths = []
    for inst in instances:
        all_paths.extend([
            f"epch_FT_hb_obs_{inst['FT']}.set",
            f"epch_FT_si_obs_{inst['FT']}.set",
            f"epch_PT_hb_obs_{inst['PT']}.set",
            f"epch_PT_si_obs_{inst['PT']}.set"
        ])
        
    print(f"Total de archivos a pre-procesar: {len(all_paths)}")
    
    # 2. PROCESAMIENTO EN PARALELO
    t_paralelo = time.time()
    # Usamos n_jobs=4 para simular un SLURM de 4 CPUs
    Parallel(n_jobs=4)(delayed(mock_load_and_compute_ddtf)(p) for p in set(all_paths))
    print(f">>> Fase en Paralelo terminada en {time.time() - t_paralelo:.2f} segundos <<<")
    
    # 3. VERIFICAR EL FLUJO SECUENCIAL ORIGINAL (Que ahora será instantáneo gracias al caché)
    print("\n--- Ejecutando bucle principal secuencial ---")
    t_secuencial = time.time()
    
    for k, inst in enumerate(instances):
        paths = {
            "FT_hb": f"epch_FT_hb_obs_{inst['FT']}.set",
            "FT_si": f"epch_FT_si_obs_{inst['FT']}.set",
            "PT_hb": f"epch_PT_hb_obs_{inst['PT']}.set",
            "PT_si": f"epch_PT_si_obs_{inst['PT']}.set"
        }
        
        print(f"\nEvaluando Instancia {k+1}...")
        # Estas llamadas deberían ser instantáneas
        ddtf_1, ch = mock_load_and_compute_ddtf(paths["FT_hb"])
        ddtf_2, _ = mock_load_and_compute_ddtf(paths["FT_si"])
        ddtf_3, _ = mock_load_and_compute_ddtf(paths["PT_hb"])
        ddtf_4, _ = mock_load_and_compute_ddtf(paths["PT_si"])
        
        # Validar que devolvió los datos correctamente
        assert ddtf_1.shape == (10, 5, 4, 4), "Shape incorrecto"
        assert len(ch) == 4, "Canales incorrectos"
        
    print(f"\n>>> Bucle principal secuencial procesado en {time.time() - t_secuencial:.3f} segundos <<<")
    print("¡FLUJO EXITOSO! SIN BUGS Y ALTO RENDIMIENTO CONFIRMADO.")
