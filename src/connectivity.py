import numpy as np
import os
from pathlib import Path

import parameters
from src.preprocessing import load_roi_epochs


def build_lagged_design(roi_epochs, p, p_start=None):
    """
    Construye las matrices de regresión de un MVAR(p) multi-ensayo (con intercepto).
    Los rezagos nunca cruzan el borde entre épocas: cada época aporta las muestras
    t = p_start, ..., n_times - 1, regresadas sobre sus propios p valores previos.

    Parámetros:
    -----------
    roi_epochs: np.ndarray (n_epochs, n_rois, n_times)
    p: orden del modelo
    p_start: primera muestra usada como respuesta en cada época (default: p).
             Fijarlo en max_p permite comparar AIC/BIC entre órdenes sobre la misma muestra.

    Retorna:
    --------
    Y: (n_obs, n_rois), X: (n_obs, 1 + n_rois * p)
    """
    p_start = p if p_start is None else p_start
    n_epochs, n_rois, n_times = roi_epochs.shape
    n_per_epoch = n_times - p_start

    Y = np.empty((n_epochs * n_per_epoch, n_rois))
    X = np.ones((n_epochs * n_per_epoch, 1 + n_rois * p))

    for ep in range(n_epochs):
        rows = slice(ep * n_per_epoch, (ep + 1) * n_per_epoch)
        x = roi_epochs[ep]
        Y[rows] = x[:, p_start:].T
        for lag in range(1, p + 1):
            X[rows, 1 + (lag - 1) * n_rois: 1 + lag * n_rois] = x[:, p_start - lag: n_times - lag].T

    return Y, X


def fit_mvar_multitrial(roi_epochs, p, p_start=None):
    """
    Ajusta por mínimos cuadrados un único MVAR(p) a todas las épocas de un registro:
        x_t = c + sum_{l=1..p} A_l x_{t-l} + e_t,   Cov(e_t) = V

    Retorna:
    --------
    dict con
        A: (p, n_rois, n_rois) coeficientes, A[l-1][dest, src]
        V: (n_rois, n_rois) covarianza residual (corregida por grados de libertad)
        sigma_ml: (n_rois, n_rois) covarianza residual de máxima verosimilitud
        n_obs: número de muestras usadas como respuesta
        max_root: módulo de la mayor raíz de la matriz compañera (< 1 => estable)
    """
    Y, X = build_lagged_design(roi_epochs, p, p_start)
    n_obs, n_rois = Y.shape

    B, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (1 + n_rois*p, n_rois)
    resid = Y - X @ B
    sigma_ml = resid.T @ resid / n_obs
    V = resid.T @ resid / (n_obs - X.shape[1])

    A = np.stack([B[1 + l * n_rois: 1 + (l + 1) * n_rois].T for l in range(p)])

    companion = np.zeros((n_rois * p, n_rois * p))
    companion[:n_rois] = np.hstack(list(A))
    companion[n_rois:, :-n_rois] = np.eye(n_rois * (p - 1))
    max_root = np.max(np.abs(np.linalg.eigvals(companion)))

    return {'A': A, 'V': V, 'sigma_ml': sigma_ml, 'n_obs': n_obs, 'max_root': max_root}


def ddtf_from_mvar(A, V, sampling_freq, freqs=None):
    """
    dDTF (Korzeniewska et al., 2003) = DTF normalizado * coherencia parcial.

    Parámetros:
    -----------
    A: (p, n_rois, n_rois) coeficientes MVAR
    V: (n_rois, n_rois) covarianza residual
    sampling_freq: frecuencia de muestreo del MVAR (Hz)
    freqs: frecuencias a evaluar (default parameters.FS_GLOBAL)

    Retorna:
    --------
    dDTF: np.ndarray (n_fs, n_rois, n_rois), dDTF[f, dest, src], diagonal en cero
    """
    if freqs is None:
        freqs = parameters.FS_GLOBAL
    p, n_rois, _ = A.shape
    eps = np.finfo(float).eps

    lags = np.arange(1, p + 1)
    exp_matrix = np.exp(-2j * np.pi * np.outer(freqs, lags) / sampling_freq)  # (n_fs, p)
    A_f = np.eye(n_rois) - np.einsum('fp,pij->fij', exp_matrix, A)
    H = np.linalg.inv(A_f)                                # (n_fs, n_rois, n_rois)
    S = H @ V @ np.conj(np.transpose(H, (0, 2, 1)))       # matriz espectral

    # DTF normalizado por fila: |H_ij|^2 / sum_m |H_im|^2
    H2 = np.abs(H) ** 2
    DTF = np.sqrt(H2 / (H2.sum(axis=2, keepdims=True) + eps))

    # Coherencia parcial: se obtiene de la matriz espectral inversa G = S^-1
    G = np.linalg.inv(S)
    G_diag = np.abs(np.diagonal(G, axis1=1, axis2=2))      # (n_fs, n_rois)
    PC = np.abs(G) / (np.sqrt(G_diag[:, :, None] * G_diag[:, None, :]) + eps)

    dDTF = DTF * PC
    dDTF[:, np.arange(n_rois), np.arange(n_rois)] = 0.0
    return np.nan_to_num(dDTF)


def load_and_compute_ddtf(filepath, p, roi_method=None):
    """
    Carga un registro (.set), ajusta un MVAR multi-ensayo sobre sus ROIs remuestreadas
    y calcula el dDTF. Resultado cacheado en data/ddtf_cache.

    Retorna:
    --------
    dict con ddtf (n_fs, n_rois, n_rois), max_root, n_obs, n_epochs
    """
    if roi_method is None:
        roi_method = parameters.ROI_EXTRACTION_METHOD

    stem = Path(filepath).stem
    cache_dir = "data/ddtf_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # 'mt' = MVAR multi-ensayo; invalida los cachés por época a 500 Hz
    cache_file = f"{cache_dir}/{stem}_fs{int(parameters.FS_TARGET)}_p{p}_{roi_method}_mt.npz"
    if os.path.exists(cache_file):
        cached = np.load(cache_file)
        return {k: cached[k] for k in cached.files}

    roi_data = load_roi_epochs(filepath, parameters.ROIS, method=roi_method)
    fit = fit_mvar_multitrial(roi_data, p)
    result = {
        'ddtf': ddtf_from_mvar(fit['A'], fit['V'], parameters.FS_TARGET),
        'max_root': np.float64(fit['max_root']),
        'n_obs': np.int64(fit['n_obs']),
        'n_epochs': np.int64(roi_data.shape[0]),
    }
    np.savez(cache_file, **result)
    return result
