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

# ==============================================================================
# PREPROCESAMIENTO
# ==============================================================================
# Los .set vienen a 500 Hz pero con un pasa-bajos previo en ~12 Hz (-65 dB a 13 Hz).
# Ajustar un MVAR a 500 Hz sobre una señal de ~12 Hz de ancho de banda deja los
# rezagos casi colineales (raíces en el círculo unitario) y una memoria de solo p * 2 ms.
# Además, cualquier tramo de banda de rechazo bajo Nyquist hace a la señal casi
# perfectamente predecible: a 50 Hz ningún MVAR con p >= 6 resultó estable y el BIC
# decrecía sin mínimo. A 20 Hz (Nyquist 10 Hz, cubre Delta y Theta) todos los modelos
# son estables y el BIC tiene un mínimo claro.
FS_ORIGINAL = 500.0
FS_TARGET = 20.0

# Las épocas son segmentos contiguos de 3 s (evento 'sg'), no bloqueados a un estímulo:
# el índice de época no tiene significado compartido entre sujetos. Por eso se ajusta
# UN MVAR por sujeto y condición usando todas las épocas (MVAR multi-ensayo).
# Cada época se des-tendencia linealmente antes del ajuste (deriva DC / sin pasa-altos).
DETREND = "linear"

# Solo bandas con señal: Beta y Gamma están en la banda de rechazo del filtro y Alpha
# en su pendiente de transición. Delta comienza en 1 Hz porque una época de 3 s apenas
# contiene 1.5 ciclos de 0.5 Hz.
F_BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
}
FREQ_STEP = 0.25
F_MIN = min([b[0] for b in F_BANDS.values()])
F_MAX = max([b[1] for b in F_BANDS.values()])
FS_GLOBAL = arange(F_MIN, F_MAX + FREQ_STEP, FREQ_STEP)

# Orden del MVAR elegido por BIC sobre el MVAR multi-ensayo
# a 20 Hz (ver scripts/find_optimal_p.py y plots/mvar_order_selection_mean.json)
P_OPTIMO = 3
MAX_P = 15

# ==============================================================================
# CONFIGURACIÓN DE REGIONS OF INTEREST (ROIs) COMO SUPER-NODOS
# ==============================================================================
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
