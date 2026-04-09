from pathlib import Path

HB_LIST = Path("data/F06_hb_sg").iterdir()
SI_LIST = Path("data/F06_sisg").iterdir()

HEARTBEAT = [str(file) for file in HB_LIST if file.suffix == ".set"]
SILENCE = [str(file) for file in SI_LIST if file.suffix == ".set"]
ELP_FILE = "data/eeglab_65chanlocs.elp"

DROPPED_CHANNELS_MATLAB = [1, 5, 8, 10, 17, 23, 29, 32, 35, 37, 39, 43, 47, 55, 61, 62, 63, 64]
DROPPED_CHANNELS = [f"E{channel}" for channel in DROPPED_CHANNELS_MATLAB]

# Añadimos Cz u otros canales extraños si es necesario descartarlos
DROPPED_CHANNELS.append("Cz")

EPOCH_DURATION_S = 0.5
EPOCH_OVERLAP_S = 0
F_BANDS = [
    (0.5, 4.0),   # Delta
    (4.0, 8.0),   # Theta
    (8.0, 13.0),  # Alpha
    (13.0, 30.0), # Beta
    (30.0, 100.0) # Gamma
]
FREQ_STEP = 0.5