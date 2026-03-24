from pathlib import Path

import mne
from mne_connectivity import EpochConnectivity
from mne_connectivity.vector_ar import vector_auto_regression


DATA_FILES = [
	Path("data/epch_si_prp_R004_6mFT.set"),
]

# Signal preprocessing (adjust to your study)
L_FREQ = 0.5
H_FREQ = 40.0

DROPPED_CHANNELS_MATLAB = [1, 5, 10, 17, 23, 29, 32, 35, 37, 39, 43, 47, 55, 61, 62, 63, 64]
DROPPED_CHANNELS = [channel - 1 for channel in DROPPED_CHANNELS_MATLAB]  # Convert to 0-based indexing

# Manual epoching parameters when the file is continuous
EPOCH_DURATION_S = 2.0
EPOCH_OVERLAP_S = 1.0

# VAR model order, obtain through AIC or other criteria
MVAR_LAGS = 10

# Manually create epochs
def create_epochs_from_raw(raw: mne.io.BaseRaw) -> mne.Epochs:
	raw = raw.copy().filter(l_freq=L_FREQ, h_freq=H_FREQ).crop(tmin=1.0)
	raw.drop_channels(DROPPED_CHANNELS)

	epochs = mne.make_fixed_length_epochs(
		raw,
		duration=EPOCH_DURATION_S,
		overlap=EPOCH_OVERLAP_S,
		preload=True
	)

	return epochs


def fit_mvar_per_epoch(epochs: mne.Epochs, lags: int = MVAR_LAGS) -> EpochConnectivity:
	data = epochs.get_data(copy=False)  # shape: (n_epochs, n_channels, n_times)

	conn = vector_auto_regression(
		data=data,
		names=epochs.ch_names,
		lags=lags,
		model="dynamic"
	)
	return conn


if __name__ == "__main__":
	for set_file in DATA_FILES:
		raw = mne.io.read_raw_eeglab(set_file, preload=True)
		epochs = create_epochs_from_raw(raw)
		conn = fit_mvar_per_epoch(epochs)

		print(
			f"{set_file.name}: "
			f"n_epochs={len(epochs)}, n_channels={len(epochs.ch_names)}, "
			f"n_times/epoch={len(epochs.times)}"
		)
		print(f"Connectivity object: {type(conn).__name__}")
		print(f"Dense shape: {conn.get_data(output='dense').shape}")