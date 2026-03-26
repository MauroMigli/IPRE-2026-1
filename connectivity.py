import time
import numpy as np
import mne
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo
from statsmodels.tsa.api import VAR
from scipy import stats

# ==========================================
# 1. PARÁMETROS Y CONSTANTES
# ==========================================
FILE_SI = "data/epch_si_prp_R004_6mFT.set"
FILE_NE = "data/epch_ne_prp_R004_6mFT.set"
ELP_FILE = "data/eeglab_65chanlocs.elp"

DROPPED_CHANNELS_MATLAB = [1, 5, 8, 10, 17, 23, 29, 32, 35, 37, 39, 43, 47, 55, 61, 62, 63, 64]
DROPPED_CHANNELS = [f"E{channel}" for channel in DROPPED_CHANNELS_MATLAB]
# Añadimos Cz u otros canales extraños si es necesario descartarlos
DROPPED_CHANNELS.append("Cz")

EPOCH_DURATION_S = 2.0
EPOCH_OVERLAP_S = 0
MVAR_LAGS = 5 
L_FREQ = 4.0
H_FREQ = 8.0
FREQ_STEP = 0.5

# ==========================================
# 2. FUNCIONES BASE
# ==========================================

def export_interactive_3d_network(coords_3d, p_values, channel_names, filename="red_conectividad_3d.html"):
    """
    Genera un archivo HTML interactivo usando Plotly para visualizar la red 3D.
    Permite rotar, hacer zoom y hover sobre los nodos.
    """
    print(f"\n--- Generando visualización interactiva en {filename} ---")
    
    p_threshold = 0.05
    highly_sig_threshold = 0.01
    n_ch = len(channel_names)
    
    # 1. Crear Nodos (Electrodos)
    xs, ys, zs = coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]
    
    nodos_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers+text',
        marker=dict(size=6, color='black', opacity=0.7),
        text=channel_names, # Nombres de los canales al hover
        textposition="top center",
        hoverinfo='text',
        name='Electrodos'
    )
    
    # 2. Crear Aristas (Flechas de Conectividad)
    edges_traces = []
    
    for i in range(n_ch):
        for j in range(n_ch):
            if i != j and p_values[i, j] < p_threshold:
                # i es destino, j es origen (flujo j -> i)
                x_src, y_src, z_src = coords_3d[j]
                x_dest, y_dest, z_dest = coords_3d[i]
                
                # Definir color y grosor según significancia
                if p_values[i, j] < highly_sig_threshold:
                    color = 'red'     # Altamente significativo (p < 0.01)
                    width = 4
                else:
                    color = 'blue'  # Moderadamente significativo (p < 0.05)
                    width = 0.4
                
                # Crear la línea de la arista
                edge_trace = go.Scatter3d(
                    x=[x_src, x_dest],
                    y=[y_src, y_dest],
                    z=[z_src, z_dest],
                    mode='lines',
                    line=dict(color=color, width=width),
                    hoverinfo='none', # No mostrar info al hover en la línea
                    showlegend=False
                )
                edges_traces.append(edge_trace)

    # 3. Configurar Layout y Escena 3D
    layout = go.Layout(
        title=f"Red de Conectividad Direccional Significativa (ANOVA p < {p_threshold})",
        scene=dict(
            xaxis=dict(title='X (cm)', showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title='Y (cm)', showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(title='Z (cm)', showgrid=False, zeroline=False, showticklabels=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)) # Posición inicial de la cámara
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # 4. Combinar y Exportar
    fig = go.Figure(data=[nodos_trace] + edges_traces, layout=layout)
    
    # Guardar como archivo HTML
    pyo.plot(fig, filename=filename, auto_open=False)
    print(f"Archivo generado exitosamente.")



def create_epochs_from_raw(raw: mne.io.BaseRaw) -> mne.Epochs:
    raw_processed = raw.copy().filter(l_freq=L_FREQ, h_freq=H_FREQ)
    raw_processed.crop(tmin=1.0)
    
    ch_to_drop = [ch for ch in DROPPED_CHANNELS if ch in raw_processed.ch_names]
    raw_processed.drop_channels(ch_to_drop)

    epochs = mne.make_fixed_length_epochs(
        raw_processed,
        duration=EPOCH_DURATION_S,
        overlap=EPOCH_OVERLAP_S,
        preload=True
    )
    return epochs

def get_3d_positions(elp_filepath, channel_names):
    """Parsea el archivo .elp y extrae las coordenadas 3D exactas para los canales."""
    with open(elp_filepath, 'r') as f:
        text = f.read()
        
    tokens = text.split()
    pos_dict = {}
    
    i = 0
    while i < len(tokens):
        if tokens[i].startswith('['): 
            i += 2
            continue
        ch_name = tokens[i]
        try:
            x, y, z = float(tokens[i+1]), float(tokens[i+2]), float(tokens[i+3])
            pos_dict[ch_name] = np.array([x, y, z])
            i += 4
        except ValueError:
            i += 1
            
    coords_3d = np.zeros((len(channel_names), 3))
    for idx, name in enumerate(channel_names):
        if name in pos_dict:
            coords_3d[idx] = pos_dict[name]
    return coords_3d

# ==========================================
# 3. MOTOR PRINCIPAL DE CONECTIVIDAD
# ==========================================
def process_dDTF(set_file):
    print(f"\nProcesando archivo: {set_file}")
    raw = mne.io.read_raw_eeglab(set_file, preload=True)
    epochs = create_epochs_from_raw(raw)
    data_epochs = epochs.get_data(copy=False)
    
    n_epochs, n_channels, _ = data_epochs.shape
    n_epochs = 10
    f_sampling = raw.info["sfreq"]
    dt = 1.0 / f_sampling
    fs = np.arange(L_FREQ, H_FREQ, FREQ_STEP)
    n_fs = len(fs)
    
    dDTF_global = np.zeros((n_epochs, n_fs, n_channels, n_channels))

    start_time = time.perf_counter()
    for epoch in range(n_epochs):
        data_ep = data_epochs[epoch] 
        print(f"Epoch: ---- {epoch} ----")


        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # 1. Ajuste del modelo Bivariado
                pair_data = np.vstack((data_ep[i], data_ep[j])).T
                model = VAR(pair_data)
                fitted = model.fit(maxlags=MVAR_LAGS)
                
                A_pair = fitted.coefs
                V_pair = fitted.sigma_u
                
                # 2. Vectorización de frecuencias (Mucho más rápido)
                k_lags = np.arange(1, MVAR_LAGS + 1)
                exp_matrix = np.exp(-2j * np.pi * np.outer(fs, k_lags) * dt)
                A_f_all = np.eye(2, dtype=complex) - np.einsum('f p, p i j -> f i j', exp_matrix, A_pair)
                
                # 3. Extracción de direccionalidad
                for f_idx in range(n_fs):
                    H_f = np.linalg.inv(A_f_all[f_idx])
                    S_f = H_f.conj() @ V_pair @ H_f.T
                    
                    row_sums = np.sum(np.abs(H_f)**2, axis=1)
                    
                    for row, col, dest, src in [(0, 1, i, j), (1, 0, j, i)]:
                        DTF = np.abs(H_f[row, col]) / np.sqrt(row_sums[row])
                        PC = np.abs(S_f[row, col]) / np.sqrt(np.abs(S_f[row, row]) * np.abs(S_f[col, col]))
                        
                        dDTF_global[epoch, f_idx, dest, src] = DTF * PC

    # Normalización del dDTF
    for epoch in range(n_epochs):
        for f_idx in range(n_fs):
            row_sq_sums = np.sum(dDTF_global[epoch, f_idx]**2, axis=1, keepdims=True)
            row_sq_sums[row_sq_sums == 0] = 1 
            dDTF_global[epoch, f_idx] /= np.sqrt(row_sq_sums)

    end_time = time.perf_counter()
    print(f"MVAR Procesado en {end_time - start_time:.2f}s")
    
    # Condensar la dimensión de frecuencia promediándola para el ANOVA
    ddtf_mean_freq = np.mean(dDTF_global, axis=1)
    
    return ddtf_mean_freq, epochs.ch_names

# ==========================================
# 4. EJECUCIÓN DEL PIPELINE Y ANOVA
# ==========================================
if __name__ == "__main__":
    ddtf_si, ch_names = process_dDTF(FILE_SI)
    ddtf_ne, _ = process_dDTF(FILE_NE)

    print("\nCalculando ANOVA sobre las épocas...")
    F_stat, p_values = stats.f_oneway(ddtf_si, ddtf_ne, axis=0)

    # ==========================================
    # 5. EXPORTAR RED 3D INTERACTIVA
    # ==========================================
    # Extraer posiciones 3D
    coords_3d = get_3d_positions(ELP_FILE, ch_names)

    # Llamar a la función de Plotly en lugar de usar Matplotlib
    export_interactive_3d_network(
        coords_3d=coords_3d, 
        p_values=p_values, 
        channel_names=ch_names, 
        filename="red_conectividad_3d.html"
    )