from pathlib import Path
from mne_connectivity.base import EpochTemporalConnectivity
from mne_connectivity.vector_ar import vector_auto_regression
import numpy as np
import numpy.typing as npt
import mne


DATA_FILES = [
	Path("data/epch_si_prp_R004_6mFT.set"),
    Path("data/epch_ne_prp_R004_6mFT.set"),
]

# Parametros
DROPPED_CHANNELS_MATLAB = [1, 5, 8, 10, 17, 23, 29, 32, 35, 37, 39, 43, 47, 55, 61, 62, 63, 64]
DROPPED_CHANNELS = [f"E{channel}" for channel in DROPPED_CHANNELS_MATLAB]  # Convert to 0-based indexing
EPOCH_DURATION_S = 2.0
EPOCH_OVERLAP_S = 1.0
MVAR_LAGS = 10
L_FREQ = 0.5
H_FREQ = 40.0


# Manually create epochs
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


def fit_mvar_per_epoch(epochs: mne.Epochs, lags: int = MVAR_LAGS) -> EpochTemporalConnectivity:
	data = epochs.get_data(copy=False)  # shape: (n_epochs, n_channels, n_times)

	conn = vector_auto_regression(
		data=data,
		names=epochs.ch_names,
		lags=lags,
		model="dynamic"
	)
	return conn


set_file = "data/epch_si_prp_R004_6mFT.set"

raw: mne.io.BaseRaw = mne.io.read_raw_eeglab(set_file, preload=True)

epochs: mne.Epochs = create_epochs_from_raw(raw)
data_epochs: npt.NDArray = epochs.get_data()
n_epochs, n_channels, n_samples = data_epochs.shape

print(
	"METADATA\n"
	f"n_epochs, n_channels, n_samples = {data_epochs.shape}\n"
	f"Sampling Freq: {raw.info["sfreq"]}\n"
	)


conn: EpochTemporalConnectivity = fit_mvar_per_epoch(epochs)

# shape = (epochs, nodes, nodes, p) -> (epochs, p, nodes, nodes)
data_model: npt.NDArray = conn.get_data(output="dense")
A = np.transpose(data_model, axes = (0, 3, 1, 2))