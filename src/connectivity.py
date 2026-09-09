import numpy as np

try:
    from statsmodels.tsa.api import VAR
except ImportError:
    VAR = None

try:
    import mne
except ImportError:
    mne = None

import time
import os
from pathlib import Path

import parameters 
from src.preprocessing import clean_epochs

def aggregate_channels_to_rois(data, ch_names, rois_dict=None, method="mean"):
    """
    Colapsa las señales multicanal a super-nodos (ROIs) anatómicamente definidos.
    
    Parámetros:
    -----------
    data: np.ndarray de forma (n_channels, n_times) o (n_epochs, n_channels, n_times)
    ch_names: lista con los nombres de los canales en 'data'
    rois_dict: dict de ROIs {nombre_roi: [canales]} (por defecto parameters.ROIS)
    method: 'mean' (promedio espacial) o 'pca' (primer componente principal espacial)
    
    Retorna:
    --------
    roi_data: np.ndarray de forma (n_rois, n_times) o (n_epochs, n_rois, n_times)
    """
    if rois_dict is None:
        rois_dict = parameters.ROIS

    is_3d = (data.ndim == 3)
    if not is_3d:
        data = data[np.newaxis, ...]  # (1, n_channels, n_times)

    n_epochs, n_channels, n_times = data.shape
    ch_to_idx = {name: idx for idx, name in enumerate(ch_names)}
    
    roi_names = list(rois_dict.keys())
    n_rois = len(roi_names)
    roi_data = np.zeros((n_epochs, n_rois, n_times), dtype=data.dtype)

    for r_idx, roi_name in enumerate(roi_names):
        member_channels = rois_dict[roi_name]
        valid_indices = [ch_to_idx[ch] for ch in member_channels if ch in ch_to_idx]

        if not valid_indices:
            continue

        if method == "pca" and len(valid_indices) > 1:
            for ep in range(n_epochs):
                cluster_signals = data[ep, valid_indices, :]  # (m, n_times)
                centered = cluster_signals - np.mean(cluster_signals, axis=1, keepdims=True)
                try:
                    u, s, vh = np.linalg.svd(centered, full_matrices=False)
                    comp = vh[0] * s[0]
                    mean_sig = np.mean(centered, axis=0)
                    if np.corrcoef(comp, mean_sig)[0, 1] < 0:
                        comp = -comp
                    roi_data[ep, r_idx, :] = comp
                except np.linalg.LinAlgError:
                    roi_data[ep, r_idx, :] = np.mean(cluster_signals, axis=0)
        else:
            roi_data[:, r_idx, :] = np.mean(data[:, valid_indices, :], axis=1)

    return roi_data if is_3d else roi_data[0]


def process_dDTF_multivariate_rois(roi_data_epochs, sampling_freq: float, p: int):
    """
    Calcula la función de transferencia dirigida directa (dDTF) multivariada
    ajustando un modelo MVAR global sobre todos los super-nodos (ROIs).
    
    Parámetros:
    -----------
    roi_data_epochs: np.ndarray (n_epochs, n_rois, n_times)
    sampling_freq: frecuencia de muestreo en Hz
    p: orden del modelo MVAR (rezago)
    
    Retorna:
    --------
    dDTF: np.ndarray (n_epochs, n_fs, n_rois, n_rois)
    """
    if VAR is None:
        raise ImportError(
            "statsmodels no está disponible en este entorno de Python.\n"
            "Por favor, activa el entorno virtual (ej: source ../.venv/bin/activate) "
            "o instala statsmodels con: pip install statsmodels"
        )

    n_epochs, n_rois, n_times = roi_data_epochs.shape
    dt = 1.0 / sampling_freq
    n_fs = len(parameters.FS_GLOBAL)
    
    dDTF = np.zeros((n_epochs, n_fs, n_rois, n_rois))
    eps = np.finfo(float).eps
    
    k_lags = np.arange(1, p + 1)
    exp_matrix = np.exp(-2j * np.pi * np.outer(parameters.FS_GLOBAL, k_lags) * dt)
    
    timer = time.time()
    for epoch in range(n_epochs):
        if epoch % 10 == 0:
            print(f"  -> Procesando MVAR Multivariado ROI Epoch {epoch + 1}/{n_epochs}... ({time.time() - timer:.2f} seg)", flush=True)
            
        data_ep = roi_data_epochs[epoch].T  # (n_times, n_rois)
        try:
            model = VAR(data_ep)
            fitted = model.fit(maxlags=p)
            A = fitted.coefs  # (p, n_rois, n_rois)
            V = fitted.sigma_u  # (n_rois, n_rois)
        except Exception:
            continue
            
        A_f_all = np.eye(n_rois, dtype=complex) - np.einsum('f p, p i j -> f i j', exp_matrix, A)
        
        for f_idx in range(n_fs):
            try:
                H_f = np.linalg.inv(A_f_all[f_idx])
            except np.linalg.LinAlgError:
                continue
                
            S_f = H_f.conj() @ V @ H_f.T
            row_sums = np.sum(np.abs(H_f)**2, axis=1)
            
            for dest in range(n_rois):
                for src in range(n_rois):
                    if dest == src:
                        continue
                        
                    den_DTF = np.sqrt(row_sums[dest])
                    DTF = np.abs(H_f[dest, src]) / (den_DTF + eps)
                    
                    den_PC = np.sqrt(np.abs(S_f[dest, dest]) * np.abs(S_f[src, src]))
                    PC = np.abs(S_f[dest, src]) / (den_PC + eps)
                    
                    if not (np.isnan(DTF) or np.isnan(PC)):
                        dDTF[epoch, f_idx, dest, src] = DTF * PC
                        
    return dDTF


def process_dDTF_global(data_epochs, sampling_freq: float, p: int):
    """
    Calcula el dDTF en BANDA ANCHA canal a canal (bivariado) con protección de estabilidad numérica (Epsilon).
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
                        den_DTF = np.sqrt(row_sums[row])
                        DTF = np.abs(H_f[row, col]) / (den_DTF + eps)
                        
                        den_PC = np.sqrt(np.abs(S_f[row, row]) * np.abs(S_f[col, col]))
                        PC = np.abs(S_f[row, col]) / (den_PC + eps)
                        
                        if np.isnan(DTF) or np.isnan(PC):
                            dDTF_global[epoch, f_idx, dest, src] = 0.0
                        else:
                            dDTF_global[epoch, f_idx, dest, src] = DTF * PC

    return dDTF_global


def load_and_compute_ddtf(filepath, p=7, sampling_freq=500.0, use_rois=True, roi_method=None):
    """
    Carga los datos limpios y calcula dDTF con un sistema de caché.
    Soporta modo ROIs super-nodos (por defecto) o cálculo canal a canal.
    """
    if roi_method is None:
        roi_method = getattr(parameters, 'ROI_EXTRACTION_METHOD', 'mean')
        
    stem = Path(filepath).stem
    cache_dir = "data/ddtf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    suffix = f"_p{p}_rois_{roi_method}" if use_rois else f"_p{p}_channels"
    cache_ddtf = f"{cache_dir}/{stem}{suffix}_ddtf.npy"
    cache_labels = f"{cache_dir}/{stem}{suffix}_labels.npy"
    
    if os.path.exists(cache_ddtf) and os.path.exists(cache_labels):
        return np.load(cache_ddtf), np.load(cache_labels, allow_pickle=True)
        
    epochs = clean_epochs(mne.io.read_epochs_eeglab(filepath, verbose=False))
    ch_names = epochs.ch_names
    data = epochs.get_data(copy=False)
    
    if use_rois:
        roi_data = aggregate_channels_to_rois(data, ch_names, parameters.ROIS, method=roi_method)
        ddtf = process_dDTF_multivariate_rois(roi_data, sampling_freq=sampling_freq, p=p)
        labels = np.array(parameters.ROI_NAMES, dtype=object)
    else:
        ddtf = process_dDTF_global(data, sampling_freq=sampling_freq, p=p)
        labels = np.array(ch_names, dtype=object)
        
    np.save(cache_ddtf, ddtf)
    np.save(cache_labels, labels)
    
    return ddtf, labels
