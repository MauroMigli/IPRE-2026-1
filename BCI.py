import numpy as np
import mne
from sys import argv
from statsmodels.tsa.api import VAR
from mne_connectivity.vector_ar import vector_auto_regression
from lag import AIC

# PENDING: Hacer interfaz con PyQt para data entry
#          y escoger epocas/bandas de frecuencia.

path = "./data/A01E.gdf"
raw = mne.io.read_raw_gdf(path, preload=True)

# 0. Datos con parpadeos y artefactos removidos

low_pass = 0.5
high_pass = 4.0

# 1. Filtrar por banda
raw = raw.copy().filter(l_freq=0.5, h_freq= 4.0)
print(raw.get_data().shape)

# 2. Epoching: ventanas de tiempo del estimulo

# events   :  np.array, [[n_muestra, basura, id_evento], ...]
# events_id: {"nombre_evento": "id"}
events, event_id = mne.events_from_annotations(raw)

# t_min: tiempo relativo antes de partida del epoch.
# t_max: lo mismo. largo = t_max - t_min
epochs = mne.Epochs(raw, 
                    events, 
                    event_id, 
                    tmin=-0.5, # En segundos
                    tmax=2.0,
                    baseline=None, 
                    preload=True,
                    event_repeated='drop')

data = epochs.get_data()
n_epochs, n_nodes, n_samples = data.shape

print("Canales:", epochs.ch_names)


id_event = {id: nombre for nombre, id in event_id.items()}

print("cantidad de eventos:", len(event_id))
print("cantidad de epochs:", n_epochs)

# 3. Obtener orden del modelo (AIC)

p = AIC(data, n_epochs, n_nodes, n_samples)

# 4. Calculo

con = vector_auto_regression(data, lags=p)

# shape = (epochs, nodes, nodes, p) -> (epochs, p, nodes, nodes)
A = np.transpose(
    con.get_data(output='dense'),
    axes = (0, 3, 1, 2)
)
f_sampling = con.attrs.get('sfreq')
dt = 1.0 / f_sampling

step = 0.1
n_fs = int((high_pass - low_pass) / step)
fs = [low_pass + i * 0.1  for i in range(n_fs)]

H = np.zeros((n_epochs, n_fs, n_nodes, n_nodes))
S = np.zeros((n_epochs, n_fs, n_nodes, n_nodes))

# 4.1 Obtener H y S

for epoch in range(n_epochs):
    # Calcular covarianza muestral de forma eficiente
    # np.cov() de una matriz
    data_epoch = data[epoch]

    X_hats = np.zeros((n_nodes, n_samples - p))
    
    for k in range(p):
        X_prevs = data_epoch[:][p - (k + 1) : n_samples - (k + 1)]
        X_hats += A[epoch][k] @ X_prevs

    Xs = data_epoch[:][p:]

    errors = Xs - X_hats
    V = np.cov(errors)

    for f in range(n_fs):
        H[epoch][f] = np.identity(n_nodes)


        for k in range(p):
            fourier = np.exp(-2 * np.pi * 1j * k * f * dt)
            H[epoch][f] += A[epoch][k] 
        
        H[epoch][f] = np.linalg.inv(H[epoch][f])
        S[epoch][f] = H[epoch][f].conj() @ V @ H[epoch][f].T

    # 4.2 DTF, PC y dDTF
    