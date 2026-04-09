from pathlib import Path
import numpy as np
import mne
import matplotlib.pyplot as plt
import parameters
import plot
from statsmodels.tsa.api import VAR
from scipy import stats

def create_epochs_from_raw(raw: mne.io.BaseRaw) -> mne.Epochs:
    """ Manualmente crea las epocas, 
        con un largo fijo de 0.5 segundos """    
    
    raw_processed = raw.copy().filter(l_freq=parameters.L_FREQ, h_freq=parameters.H_FREQ)
    raw_processed.crop(tmin=1.0)
    
    ch_to_drop = [ch for ch in parameters.DROPPED_CHANNELS if ch in raw_processed.ch_names]
    raw_processed.drop_channels(ch_to_drop)

    epochs = mne.make_fixed_length_epochs(
        raw_processed,
        duration=parameters.EPOCH_DURATION_S,
        overlap=parameters.EPOCH_OVERLAP_S,
        preload=True
    )
    return epochs

def p_histogram(data_epochs, epochs_lim: int, p_max=25):
    """ Grafica un histograma sobre todas las epocas
        y sobre todos los modelos para cada (i, j) nodos 
        distintos (46 * 45) / 2 en total. """ 
    
    n_epochs, n_channels, n_samples = data_epochs.shape
    
    n_epochs = max(min(n_epochs, epochs_lim), 0)
    # Lista plana para guardar cada 'p' óptimo encontrado individualmente
    optimal_ps = []

    print(f"Calculando rezagos óptimos (BIC) hasta p_max={p_max}...")
    
    for epoch in range(n_epochs):
        print(f"Epoch: ---- {epoch + 1}/{n_epochs} ----")
        data_ep = data_epochs[epoch]

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                pair_data = np.vstack((data_ep[i], data_ep[j])).T
                model = VAR(pair_data)
                
                try:
                    order_selection = model.select_order(maxlags=p_max)
                    
                    best_p = order_selection.bic
                    
                    if best_p == 0:
                        best_p = 1
                        
                    optimal_ps.append(best_p)
                except ValueError:
                    continue
    plt.figure(figsize=(10, 6))
    
    bins = np.arange(1, p_max + 2) - 0.5
    plt.hist(optimal_ps, bins=bins, edgecolor='black', alpha=0.75, color='steelblue')
    
    moda = stats.mode(optimal_ps, keepdims=True)[0][0] if optimal_ps else 0
    media = np.mean(optimal_ps) if optimal_ps else 0
    
    plt.axvline(moda, color='red', linestyle='dashed', linewidth=2, label=f'Moda (Voto mayoritario): {moda}')
    plt.axvline(media, color='green', linestyle='dashed', linewidth=2, label=f'Media: {media:.2f}')
    
    plt.title(f'Distribución de Rezagos Óptimos (p) según BIC\nTotal de modelos bivariados evaluados: {len(optimal_ps)}')
    plt.xlabel('Orden del Modelo (p)')
    plt.ylabel('Frecuencia (Cantidad de Pares x Épocas)')
    plt.xticks(range(1, p_max + 1))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    return optimal_ps        



def process_dDTF(data_epochs, sampling_freq: float, epochs_lim: int, p: int):
    n_epochs, n_channels, _ = data_epochs.shape
    n_epochs = max(min(n_epochs, epochs_lim), 0)
    
    dt = 1.0 / sampling_freq
    fs = np.arange(parameters.L_FREQ, parameters.H_FREQ, parameters.FREQ_STEP)
    n_fs = len(fs)
    
    dDTF_global = np.zeros((n_epochs, n_fs, n_channels, n_channels))

    for epoch in range(n_epochs):
        data_ep = data_epochs[epoch] 
        print(f"Epoch: ---- {epoch} ----")


        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # 1. Ajuste del modelo Bivariado
                pair_data = np.vstack((data_ep[i], data_ep[j])).T
                model = VAR(pair_data)
                fitted = model.fit(maxlags=p)
                
                A_pair = fitted.coefs
                V_pair = fitted.sigma_u
                
                # 2. Vectorización de frecuencias (Mucho más rápido)
                k_lags = np.arange(1, p + 1)
                exp_matrix = np.exp(-2j * np.pi * np.outer(fs, k_lags) * dt)
                A_f_all = np.eye(2, dtype=complex) - np.einsum('f p, p i j -> f i j', exp_matrix, A_pair)
                
                # 3. Extracción de direccionalidad
                for f_idx in range(n_fs):
                    H_f = np.linalg.inv(A_f_all[f_idx])
                    S_f = H_f.conj() @ V_pair @ H_f.T
                    
                    row_sums = np.sum(np.abs(H_f)**2, axis=1)
                    
                    for row, col, dest, src in [(0, 1, i, j), (1, 0, j, i)]:
                        DTF = np.abs(H_f[row, col]) / np.sqrt(row_sums[row])
                        PC = np.abs(S_f[row, col]) / np.sqrt(np.abs(S_f[row, row]) * np.abs(S_f[col, col]))
                        
                        dDTF_global[epoch, f_idx, dest, src] = DTF * PC

    return dDTF_global


if __name__ == "__main__":
    Path("plots").mkdir(exist_ok=True)
    
    # Asumimos que las listas HEARTBEAT y SILENCE están ordenadas y corresponden al mismo infante
    for hb_file, si_file in zip(parameters.HEARTBEAT, parameters.SILENCE):
        print(f"\nProcesando par: {Path(hb_file).name} vs {Path(si_file).name}")
        
        epochs_hb = mne.io.read_epochs_eeglab(hb_file)
        epochs_si = mne.io.read_epochs_eeglab(si_file)
        
        ch_names = epochs_hb.ch_names
        sf = epochs_hb.info['sfreq']
        coords_3d = plot.get_3d_positions(parameters.ELP_FILE, ch_names)

        for L_FREQ, H_FREQ in parameters.F_BANDS:
            print(f"\n--- Banda de frecuencia: {L_FREQ} - {H_FREQ} Hz ---")
            
            # Es necesario actualizar variables globales si process_dDTF y p_histogram las usan
            parameters.L_FREQ = L_FREQ
            parameters.H_FREQ = H_FREQ

            filtered_hb = epochs_hb.copy().filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)
            filtered_si = epochs_si.copy().filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)

            data_epochs_hb = filtered_hb.get_data(copy=False)
            data_epochs_si = filtered_si.get_data(copy=False)

            # Obtener lag óptimo (computado en un subconjunto de épocas)
            opt_ps_hb = p_histogram(data_epochs_hb, epochs_lim=3)
            opt_ps_si = p_histogram(data_epochs_si, epochs_lim=3)
            
            p_hb = int(stats.mode(opt_ps_hb, keepdims=True)[0][0]) if opt_ps_hb else parameters.MVAR_LAGS
            p_si = int(stats.mode(opt_ps_si, keepdims=True)[0][0]) if opt_ps_si else parameters.MVAR_LAGS
            
            print(f"p óptimo -> HB: {p_hb}, SI: {p_si}")

            # Calcular dDTF en todas las épocas usando el p óptimo
            print("Calculando dDTF para Heartbeat...")
            dDTF_hb = process_dDTF(data_epochs_hb, sampling_freq=sf, epochs_lim=data_epochs_hb.shape[0], p=p_hb)
            
            print("Calculando dDTF para Silencio...")
            dDTF_si = process_dDTF(data_epochs_si, sampling_freq=sf, epochs_lim=data_epochs_si.shape[0], p=p_si)

            # Promediar sobre el eje de frecuencias (axis=1) para obtener la media de la banda por época
            mean_band_dDTF_hb = np.mean(dDTF_hb, axis=1) # Shape: (n_epochs, n_channels, n_channels)
            mean_band_dDTF_si = np.mean(dDTF_si, axis=1)

            # T-test independiente sobre el eje de las épocas (axis=0)
            print("Ejecutando T-Test comparativo (HB vs SI)...")
            t_stat, p_values = stats.ttest_ind(mean_band_dDTF_hb, mean_band_dDTF_si, axis=0, equal_var=False)

            # Exportar la visualización
            filename = f"plots/net_{Path(hb_file).stem}_band_{L_FREQ}-{H_FREQ}.html"
            plot.export_interactive_3d_network(coords_3d, p_values, ch_names, filename)