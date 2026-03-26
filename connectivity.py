from pathlib import Path
import numpy as np
import numpy.typing as npt
import mne
import matplotlib.pyplot as plt
import time
from statsmodels.tsa.api import VAR

DATA_FILES = [
    Path("data/epch_si_prp_R004_6mFT.set"),
    Path("data/epch_ne_prp_R004_6mFT.set"),
]

# Parametros
DROPPED_CHANNELS_MATLAB = [1, 5, 8, 10, 17, 23, 29, 32, 35, 37, 39, 43, 47, 55, 61, 62, 63, 64]
DROPPED_CHANNELS = [f"E{channel}" for channel in DROPPED_CHANNELS_MATLAB]
EPOCH_DURATION_S = 2.0
EPOCH_OVERLAP_S = 1.0

# Asumimos el orden del modelo global validado (A optimizar posteriormente con la moda del AIC)
MVAR_LAGS = 5 
L_FREQ = 0.5
H_FREQ = 40.0
FREQ_STEP = 0.5

def create_epochs_from_raw(raw: mne.io.BaseRaw) -> mne.Epochs:
    raw_processed = raw.copy().filter(l_freq=L_FREQ, h_freq=H_FREQ)
    raw_processed.crop(tmin=1.0)
    raw_processed.drop_channels(DROPPED_CHANNELS)

    epochs = mne.make_fixed_length_epochs(
        raw_processed,
        duration=EPOCH_DURATION_S,
        overlap=EPOCH_OVERLAP_S,
        preload=True
    )
    return epochs

# 1. Cargar Datos
set_file = "data/epch_si_prp_R004_6mFT.set"
raw = mne.io.read_raw_eeglab(set_file, preload=True)
epochs = create_epochs_from_raw(raw)
data_epochs = epochs.get_data()

n_epochs, n_channels, n_samples = data_epochs.shape
f_sampling = raw.info["sfreq"]
dt = 1.0 / f_sampling

# 2. Configurar el dominio de la frecuencia
fs = np.arange(L_FREQ, H_FREQ, FREQ_STEP)
n_fs = len(fs)

print("Iniciando modelado por pares (Pairwise MVAR)...")
print(f"Épocas: {n_epochs} | Canales: {n_channels} | Frecuencias a evaluar: {n_fs}")

n_channels = 30
n_epochs = 1

# 3. Matriz global ensamblada: (épocas, frecuencias, canal_destino, canal_origen)
dDTF_global = np.zeros((n_epochs, n_fs, n_channels, n_channels))


for epoch in range(n_epochs):
    start_time = time.perf_counter()
    print(f"ESTAMOS EN EPOCH {epoch}")
    data_ep = data_epochs[epoch] # shape: (n_channels, n_samples)
    
    # Evaluar todas las combinaciones de pares posibles
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            pair_data = np.vstack((data_ep[i], data_ep[j])).T
            model = VAR(pair_data)
            fitted = model.fit(maxlags=MVAR_LAGS)
            
            # A_pair: Coeficientes de shape (p, 2, 2)
            # V_pair: Matriz de covarianza de ruido (V_epoch_i_j) de shape (2, 2)
            A_pair = fitted.coefs
            V_pair = fitted.sigma_u
            
            for f_idx, freq in enumerate(fs):
                k_lags = np.arange(1, MVAR_LAGS + 1)
                fourier = np.exp(-2j * np.pi * k_lags * freq * dt) # vector de tamaño p
                
                A_f = np.eye(2, dtype=complex) - np.sum(A_pair * fourier[:, None, None], axis=0)
                
                H_f = np.linalg.inv(A_f)
                
                S_f = H_f @ V_pair @ H_f.conj().T
                
                row_sums = np.sum(np.abs(H_f)**2, axis=1)
                
                for row, col, ch_dest, ch_src in [(0, 1, i, j), (1, 0, j, i)]:
                    DTF = np.abs(H_f[row, col]) / np.sqrt(row_sums[row])
                    PC = np.abs(S_f[row, col]) / np.sqrt(np.abs(S_f[row, row]) * np.abs(S_f[col, col]))
                    
                    dDTF_global[epoch, f_idx, ch_dest, ch_src] = DTF * PC
    
    end_time = time.perf_counter()
    print(f"Tiempo total: {start_time - end_time}s")

for epoch in range(n_epochs):
    for f_idx in range(n_fs):
        row_sq_sums = np.sum(dDTF_global[epoch, f_idx]**2, axis=1, keepdims=True)
        row_sq_sums[row_sq_sums == 0] = 1 
        dDTF_global[epoch, f_idx] /= np.sqrt(row_sq_sums)

plt.figure(figsize=(8, 6))
plt.imshow(dDTF_global[0, 0], cmap='viridis', aspect='auto') 
plt.colorbar(label='Fuerza de Conectividad (dDTF)')
plt.title(f'Matriz de Adyacencia Global (Epoch 0, f = {fs[0]}Hz)')
plt.xlabel('Canal de Origen (j)')
plt.ylabel('Canal de Destino (i)')
plt.show()