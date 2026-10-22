import numpy as np
from scipy.signal import detrend, resample_poly
try:
    import mne
except ImportError:
    mne = None
import parameters
from pathlib import Path
from fractions import Fraction

def clean_epochs(epochs):
    """
    Remove any channels listed in parameters.DROPPED_CHANNELS that exist in the epochs.
    """
    chans_in_data = epochs.ch_names
    chans_to_drop = [ch for ch in parameters.DROPPED_CHANNELS if ch in chans_in_data]
    if chans_to_drop:
        epochs.drop_channels(chans_to_drop)
    return epochs

def get_valid_subjects():
    """
    Parse parameters.HEARTBEAT and parameters.SILENCE to pair up the subjects
    that have both conditions. Returns a dictionary mapping kid_id -> info.
    """
    subjects = {}

    # Process heartbeat files
    for f in parameters.HEARTBEAT:
        stem = Path(f).stem
        parts = stem.split('_')
        if len(parts) >= 5:
            group, cond, kid_id = parts[1], parts[2], parts[4]
            if kid_id not in subjects:
                subjects[kid_id] = {'group': group, 'hb': None, 'si': None}
            if cond == 'hb':
                subjects[kid_id]['hb'] = f

    # Process silence files
    for f in parameters.SILENCE:
        stem = Path(f).stem
        parts = stem.split('_')
        if len(parts) >= 5:
            group, cond, kid_id = parts[1], parts[2], parts[4]
            if kid_id not in subjects:
                subjects[kid_id] = {'group': group, 'hb': None, 'si': None}
            if cond == 'si':
                subjects[kid_id]['si'] = f

    # Keep only subjects that have both conditions
    valid_subjects = {k: v for k, v in subjects.items() if v['hb'] and v['si']}
    return dict(sorted(valid_subjects.items()))


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


def downsample_epochs(data, fs_in=None, fs_out=None):
    """
    Remuestreo polifásico con filtro anti-aliasing a lo largo del último eje.
    500 Hz -> 50 Hz equivale a quedarse con 1 de cada 10 muestras tras filtrar.
    """
    fs_in = fs_in or parameters.FS_ORIGINAL
    fs_out = fs_out or parameters.FS_TARGET
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    if ratio == 1:
        return data
    return resample_poly(data, ratio.numerator, ratio.denominator, axis=-1)


def load_roi_epochs(filepath, rois_dict=None, method="mean"):
    """
    Pipeline de preprocesamiento de un archivo .set:
    canales podados -> super-nodos ROI -> des-tendencia por época -> remuestreo a FS_TARGET.

    Retorna:
    --------
    roi_data: np.ndarray (n_epochs, n_rois, n_times_resampled)
    """
    epochs = clean_epochs(mne.io.read_epochs_eeglab(filepath, verbose=False))
    fs_in = epochs.info['sfreq']
    data = epochs.get_data(copy=False)

    roi_data = aggregate_channels_to_rois(data, epochs.ch_names, rois_dict, method=method)
    if parameters.DETREND:
        roi_data = detrend(roi_data, axis=-1, type=parameters.DETREND)
    return downsample_epochs(roi_data, fs_in=fs_in)
