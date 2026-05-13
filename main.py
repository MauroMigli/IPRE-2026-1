import mne
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
import numpy as np

import plot
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

    # 0. Para revisar la cantidad de epocas en cada cuestion
    #    Hay cantidad dispar entre ambos archivos multiples veces.
    #for hb_file, si_file in zip(parameters.HEARTBEAT, parameters.SILENCE):
    #    epochs_hb = mne.io.read_epochs_eeglab(hb_file)
    #    epochs_si = mne.io.read_epochs_eeglab(si_file)
    #    print("--- ",len(epochs_hb), "vs", len(epochs_si), " ---")

    for hb_file, si_file in zip(parameters.HEARTBEAT, parameters.SILENCE):        
        kid_id = Path(hb_file).stem.split('_')[0]
        print(f"\n=============================================")
        print(f"Procesando Sujeto: {kid_id}")
        print(f"Archivos: {Path(hb_file).name} vs {Path(si_file).name}")
        
        # 1. Cargar datos
        epochs_hb = mne.io.read_epochs_eeglab(hb_file)
        epochs_si = mne.io.read_epochs_eeglab(si_file)
        
        # 2. Limpiar canales
        epochs_hb = clean_epochs(epochs_hb)
        epochs_si = clean_epochs(epochs_si)
        

        # Note: Esto probablemente no ayuda.
        #NUEVA_FRECUENCIA = 500.0  
        #print(f"  -> Realizando downsampling a {NUEVA_FRECUENCIA} Hz...")
        #epochs_hb = epochs_hb.resample(sfreq=NUEVA_FRECUENCIA)
        #epochs_si = epochs_si.resample(sfreq=NUEVA_FRECUENCIA)

        ch_names = epochs_hb.ch_names
        sf = epochs_hb.info['sfreq']
        
        #coords_3d = plot.get_3d_positions(parameters.ELP_FILE, ch_names)
        
        # Extraer matrices numéricas
        data_epochs_hb = epochs_hb.get_data(copy=False)
        data_epochs_si = epochs_si.get_data(copy=False)
        
        # Inicio: Esta parte se demora 13.2 minutos
        print(f"Empezando a procesar {kid_id} HB:")
        dDTF_hb_global = process_dDTF_global(data_epochs_hb, sampling_freq=sf, p=parameters.P_OPTIMO)

        print(f"Empezando a procesar {kid_id} SI:")
        dDTF_si_global = process_dDTF_global(data_epochs_si, sampling_freq=sf, p=parameters.P_OPTIMO)
        # Fin.

        for FREQ_NAME, (L_FREQ, H_FREQ) in parameters.F_BANDS.items():
            for e in range(len(epochs_hb)):
                f_indices = np.where((parameters.FS_GLOBAL >= L_FREQ) & (parameters.FS_GLOBAL < H_FREQ))[0]
                dDTF_hb_band_e = dDTF_hb_global[e, f_indices, :, :]  
                dDTF_si_band_e = dDTF_si_global[e, f_indices, :, :]
                min_epochs = min(len(dDTF_hb_band_e), len(dDTF_si_band_e))

                t_stat, p_values = stats.ttest_rel(dDTF_hb_band_e[:min_epochs], dDTF_si_band_e[:min_epochs], axis=0)
                
                np.save(f'./p_values/{kid_id}_{FREQ_NAME}.npy', p_values)