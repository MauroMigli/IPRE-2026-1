from pathlib import Path
from numpy import arange

HB_DIR = Path("data/epch_heartbeat")
SI_DIR = Path("data/epch_silence")

HEARTBEAT = [str(file) for file in HB_DIR.iterdir() if file.suffix == ".set"] if HB_DIR.exists() else []
SILENCE = [str(file) for file in SI_DIR.iterdir() if file.suffix == ".set"] if SI_DIR.exists() else []
ELP_FILE = "data/eeglab_65chanlocs.elp"

DROPPED_CHANNELS_MATLAB = [1, 5, 8, 10, 17, 23, 29, 32, 35, 37, 39, 43, 47, 55, 61, 62, 63, 64]
DROPPED_CHANNELS = [f"E{channel}" for channel in DROPPED_CHANNELS_MATLAB]

# Añadimos Cz u otros canales extraños si es necesario descartarlos
DROPPED_CHANNELS.append("Cz")

EPOCH_DURATION_S = 0.5
EPOCH_OVERLAP_S = 0
F_BANDS = {
    "Delta": (0.5, 4.0),   # Delta
    "Theta": (4.0, 8.0),   # Theta
    "Alpha": (8.0, 12.0),  # Alpha
    "Beta": (12.0, 30.0), # Beta
    "Gamma": (30.0, 100.0) # Gamma
}
FREQ_STEP = 0.5
F_MIN = min([b[0] for b in F_BANDS.values()])
F_MAX = max([b[1] for b in F_BANDS.values()])
FS_GLOBAL = arange(F_MIN, F_MAX + FREQ_STEP, FREQ_STEP)

# Orden óptimo del modelo MVAR determinado empíricamente mediante model_order.py
# (Criterio del codo / Elbow criterion sobre las curvas de información AIC y BIC)
P_OPTIMO = 5

# ==============================================================================
# CONFIGURACIÓN DE REGIONS OF INTEREST (ROIs) COMO SUPER-NODOS
# ==============================================================================
USE_ROIS = True
ROI_EXTRACTION_METHOD = "mean"  # Opciones: 'mean' (promedio espacial) o 'pca' (primer componente principal)
R_ROI_DEFAULT = 9.5  # Radio espacial (cm) sugerido para TFCE entre centroides de ROIs

# Definición formal de las 8 ROIs simétricas (cubren los canales supervivientes)
ROIS = {
    "Frontal_Medial": ["E3", "E4", "E6", "E8", "E9", "E12", "E60"],
    "Frontal_Lateral_L": ["E11", "E13", "E14", "E15", "E18", "E19"],
    "Frontal_Lateral_R": ["E2", "E54", "E56", "E57", "E58", "E59"],
    "Central_Motor_L": ["E7", "E16", "E20", "E21"],
    "Central_Motor_R": ["E41", "E50", "E51", "E53"],
    "TemporoParietal_L": ["E22", "E24", "E25", "E26", "E27", "E28", "E30"],
    "TemporoParietal_R": ["E42", "E44", "E45", "E46", "E48", "E49", "E52"],
    "Parieto_Occipital": ["E31", "E33", "E34", "E36", "E38", "E40"],
}
ROI_NAMES = list(ROIS.keys())