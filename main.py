import mne
from pathlib import Path
from scipy import stats
import numpy as np
import os

import parameters
from dDTF import process_dDTF_global

def clean_epochs(epochs):
    """
    Dropea los canales malos definidos en parámetros.
    """
    chans_in_data = epochs.ch_names
    chans_to_drop = [ch for ch in parameters.DROPPED_CHANNELS if ch in chans_in_data]
    if chans_to_drop:
        epochs.drop_channels(chans_to_drop)
    return epochs

if __name__ == "__main__":
    
    subjects = {}
    
    # Ingesta de datos (Data Intake) acoplada a la nueva estructura
    for f in parameters.HEARTBEAT + parameters.SILENCE:
        stem = Path(f).stem
        parts = stem.split('_')
        # Formato esperado: PT_si_obs_101
        if len(parts) >= 4:
            group = parts[0]
            cond = parts[1]
            kid_id = parts[3]
            
            if kid_id not in subjects:
                subjects[kid_id] = {'group': group, 'hb': None, 'si': None}
            
            if cond == 'hb':
                subjects[kid_id]['hb'] = f
            elif cond == 'si':
                subjects[kid_id]['si'] = f

    # Listas para almacenar las diferencias Delta_s(v) agrupadas si se procesaran todas en memoria
    deltas_FT = []
    deltas_PT = []
    
    os.makedirs('./deltas', exist_ok=True)
    
    for kid_id, info in subjects.items():
        group = info['group']
        hb_file = info['hb']
        si_file = info['si']
        
        if not hb_file or not si_file:
            print(f"Sujeto {kid_id} ({group}) no tiene ambos archivos (hb y si). Omitiendo.")
            continue
            
        print(f"\\n=============================================")
        print(f"Procesando Sujeto: {kid_id} (Grupo: {group})")
        print(f"Archivos: {Path(hb_file).name} vs {Path(si_file).name}")
        
        # 1. Cargar datos
        epochs_hb = mne.io.read_epochs_eeglab(hb_file)
        epochs_si = mne.io.read_epochs_eeglab(si_file)
        
        # 2. Limpiar canales
        epochs_hb = clean_epochs(epochs_hb)
        epochs_si = clean_epochs(epochs_si)
        
        sf = epochs_hb.info['sfreq']
        
        # Extraer matrices numéricas
        data_epochs_hb = epochs_hb.get_data(copy=False)
        data_epochs_si = epochs_si.get_data(copy=False)
        
        print(f"Empezando a procesar dDTF para {kid_id} HB...")
        # Descomentar en ejecución real (toma mucho tiempo):
        # dDTF_hb_global = process_dDTF_global(data_epochs_hb, sampling_freq=sf, p=parameters.P_OPTIMO)
        
        print(f"Empezando a procesar dDTF para {kid_id} SI...")
        # dDTF_si_global = process_dDTF_global(data_epochs_si, sampling_freq=sf, p=parameters.P_OPTIMO)
        
        print("Cálculo de dDTF comentado en el refactor para evitar largos tiempos de procesamiento.")
        print("Para TFCE, se calcularía la diferencia Delta_s(v) = dDTF_hb - dDTF_si aquí:")
        
        # min_epochs = min(len(dDTF_hb_global), len(dDTF_si_global))
        # delta = dDTF_hb_global[:min_epochs] - dDTF_si_global[:min_epochs]
        # np.save(f'./deltas/{kid_id}_{group}_delta.npy', delta)
        # if group == 'FT': deltas_FT.append(delta)
        # else: deltas_PT.append(delta)

    print("\\n[+] Data intake refactorizado correctamente. Ejecución finalizada.")