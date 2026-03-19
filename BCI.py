import numpy as np
import mne
import matplotlib.pyplot as plt
from sys import argv
from statsmodels.tsa.api import VAR
from mne_connectivity.vector_ar import vector_auto_regression
from utils import AIC, V_epoch

# PENDING: Hacer interfaz para data entry (?)
#          y escoger epocas/bandas de frecuencia.


path = "./data/A01E.gdf"
raw = mne.io.read_raw_gdf(path, preload=True)


# 1. Filtrar por banda
high_pass = 0.5
low_pass = 4.0

raw = raw.copy().filter(l_freq=high_pass, h_freq=low_pass)


# 2. Epoching: ventanas de tiempo del estimulo

# events   :  np.array, [[idx_muestra, basura, id_evento], ...]
# events_id: {"nombre_evento": "id"}
events, event_id = mne.events_from_annotations(raw)

# t_min: tiempo relativo antes de partida del epoch.
# t_max: lo mismo. largo = t_max - t_min
epochs = mne.Epochs(raw, 
                    events, 
                    event_id, 
                    tmin=-0.5,
                    tmax=2.0,
                    baseline=None, 
                    preload=True,
                    event_repeated='drop')

data = epochs.get_data()
n_epochs, n_nodes, n_samples = data.shape


# 3. Obtener orden del modelo (AIC)
p = 20 #AIC(data, n_epochs, n_nodes, n_samples)

# 4. Calculo matrices de conectividad
con = vector_auto_regression(data, lags=p)

# shape = (epochs, nodes, nodes, p) -> (epochs, p, nodes, nodes)
A = np.transpose(
    con.get_data(output='dense'),
    axes = (0, 3, 1, 2))

f_sampling = epochs.info['sfreq']
dt = 1.0 / f_sampling

step = 0.1
n_fs = int((low_pass - high_pass) / step)
fs = [high_pass + i * 0.1  for i in range(n_fs)]

H = np.zeros((n_epochs, n_fs, n_nodes, n_nodes), dtype=complex)
S = np.zeros((n_epochs, n_fs, n_nodes, n_nodes), dtype=complex)
dDTF = np.zeros((n_epochs, n_fs, n_nodes, n_nodes))

# 4.1 Obtener H y S
# PENDING: En vez de hacer analisis para todas las epocas 
#          simultaneamente, escoger un subconjunto (?)

for epoch in range(1): # range(epochs):
    V = V_epoch(data, A, p, epoch, n_nodes, n_samples)

    for f in range(n_fs):
        A_f = np.identity(n_nodes, dtype=complex)

        for k in range(p):
            fourier = np.exp(-2 * np.pi * 1j * (k + 1) * fs[f] * dt)
            A_f -= A[epoch][k] * fourier
        
        H[epoch][f] = np.linalg.inv(A_f)
        S[epoch][f] = H[epoch][f].conj() @ V @ H[epoch][f].T


        # 4.2 DTF, PC y dDTF
        for i in range(n_nodes):
            H_i_sum = 0
            for j in range(n_nodes):
                H_i_sum += abs(H[epoch][f][i][j]) ** 2
            
            for j in range(n_nodes):
                H_ef = H[epoch][f]
                S_ef = S[epoch][f]

                DTF = abs(H_ef[i][j]) / np.sqrt(H_i_sum)
                PC = np.sqrt(abs(S_ef[i][j]) ** 2 / (S_ef[i][i] * S_ef[j][j]))
                dDTF[epoch][f][i][j] =  DTF * PC


    # 5. Plot basico

    for f in range(n_fs):

        plt.figure(figsize=(8, 6))
        plt.imshow(dDTF[epoch][f], cmap='viridis', aspect='auto') 

        plt.colorbar(label='Fuerza de Conectividad (DTF)')
        plt.title(f'Mapa de calor de Conectividad (f = {fs[f]})')
        plt.xlabel('Canal de Origen')
        plt.ylabel('Canal de Destino')
        plt.show()