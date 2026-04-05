# Parametros usados en main.py

FILE_SI = "data/epch_si_prp_R004_6mFT.set"
FILE_NE = "data/epch_ne_prp_R004_6mFT.set"
ELP_FILE = "data/eeglab_65chanlocs.elp"

DROPPED_CHANNELS_MATLAB = [1, 5, 8, 10, 17, 23, 29, 32, 35, 37, 39, 43, 47, 55, 61, 62, 63, 64]
DROPPED_CHANNELS = [f"E{channel}" for channel in DROPPED_CHANNELS_MATLAB]

# Añadimos Cz u otros canales extraños si es necesario descartarlos
DROPPED_CHANNELS.append("Cz")

EPOCH_DURATION_S = 0.5
EPOCH_OVERLAP_S = 0
L_FREQ = 4.0
H_FREQ = 8.0
FREQ_STEP = 0.5