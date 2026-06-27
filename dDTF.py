import numpy as np
import matplotlib
matplotlib.use('Agg')
from statsmodels.tsa.api import VAR

import time
import parameters 

def process_dDTF_global(data_epochs, sampling_freq: float, p: int):
    """
    Calcula el dDTF en BANDA ANCHA con protección de estabilidad numérica (Epsilon).
    """
    n_epochs, n_channels, _ = data_epochs.shape
    dt = 1.0 / sampling_freq
    
    n_fs = len(parameters.FS_GLOBAL)
    
    dDTF_global = np.zeros((n_epochs, n_fs, n_channels, n_channels))
    eps = np.finfo(float).eps # Protección contra divisiones por cero

    timer = time.time()
    for epoch in range(n_epochs):
        if epoch % 10 == 0:
            print(f"  -> Procesando Epoch {epoch + 1}/{n_epochs}... ({time.time() - timer:.2f} seg)", flush=True)
        
        data_ep = data_epochs[epoch] 

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                pair_data = np.vstack((data_ep[i], data_ep[j])).T
                model = VAR(pair_data)
                
                try:
                    fitted = model.fit(maxlags=p)
                    A_pair = fitted.coefs
                    V_pair = fitted.sigma_u
                except Exception:
                    continue
                
                k_lags = np.arange(1, p + 1)
                exp_matrix = np.exp(-2j * np.pi * np.outer(parameters.FS_GLOBAL, k_lags) * dt)
                A_f_all = np.eye(2, dtype=complex) - np.einsum('f p, p i j -> f i j', exp_matrix, A_pair)
                
                for f_idx in range(n_fs):
                    try:
                        H_f = np.linalg.inv(A_f_all[f_idx])
                    except np.linalg.LinAlgError:
                        continue 
                        
                    S_f = H_f.conj() @ V_pair @ H_f.T
                    row_sums = np.sum(np.abs(H_f)**2, axis=1)
                    
                    for row, col, dest, src in [(0, 1, i, j), (1, 0, j, i)]:
                        # Cálculo protegido con epsilon
                        den_DTF = np.sqrt(row_sums[row])
                        DTF = np.abs(H_f[row, col]) / (den_DTF + eps)
                        
                        den_PC = np.sqrt(np.abs(S_f[row, row]) * np.abs(S_f[col, col]))
                        PC = np.abs(S_f[row, col]) / (den_PC + eps)
                        
                        if np.isnan(DTF) or np.isnan(PC):
                            dDTF_global[epoch, f_idx, dest, src] = 0.0
                        else:
                            dDTF_global[epoch, f_idx, dest, src] = DTF * PC

    return dDTF_global
