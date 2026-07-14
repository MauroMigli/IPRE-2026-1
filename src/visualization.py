import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo
import parameters

# ==========================================
# PLOT 1: Conteo de Aristas (Naive vs E[FP])
# ==========================================
def plot_edge_counts(epochs_x, naive_counts, fdr_counts, tfce_counts, expected_fp, band_name, output_dir="plots"):
    """
    Genera el gráfico de líneas mostrando las aristas que sobreviven a Naive, FDR y TFCE,
    comparado contra la esperanza matemática de falsos positivos (E[FP]).
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs_x, naive_counts, marker='o', label='Naive (p < 0.05)', color='gray', alpha=0.7)
    plt.plot(epochs_x, fdr_counts, marker='s', label='FDR (q < 0.05)', color='blue')
    plt.plot(epochs_x, tfce_counts, marker='^', label='TFCE (p < 0.05)', color='green')
    
    # Línea teórica de Falsos Positivos
    plt.axhline(y=expected_fp, color='red', linestyle='--', label=f'Esperanza FP (E[FP] = {expected_fp:.1f})')
    
    plt.title(f'Evolución temporal de aristas significativas - Banda {band_name}')
    plt.xlabel('Época (Tiempo)')
    plt.ylabel('Cantidad de Aristas Significativas')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    filename = os.path.join(output_dir, f'edge_counts_{band_name}.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


# ==========================================
# PLOT 2: Histogramas AIC y BIC
# ==========================================
def plot_aic_bic_histograms(aic_votes, bic_votes, output_dir="plots"):
    """
    Genera histogramas con la distribución de los rezagos óptimos (p) elegidos
    por AIC y BIC a través de todos los modelos bivariados.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # AIC
    axes[0].hist(aic_votes, bins=np.arange(1.5, 21.5, 1), color='skyblue', edgecolor='black')
    axes[0].set_title('Distribución de Rezagos Óptimos - AIC')
    axes[0].set_xlabel('Rezago (p)')
    axes[0].set_ylabel('Frecuencia (Votos)')
    axes[0].set_xticks(range(2, 21, 2))
    axes[0].grid(axis='y', alpha=0.75)
    
    # BIC
    axes[1].hist(bic_votes, bins=np.arange(1.5, 21.5, 1), color='lightgreen', edgecolor='black')
    axes[1].set_title('Distribución de Rezagos Óptimos - BIC')
    axes[1].set_xlabel('Rezago (p)')
    axes[1].set_ylabel('Frecuencia (Votos)')
    axes[1].set_xticks(range(2, 21, 2))
    axes[1].grid(axis='y', alpha=0.75)
    
    filename = os.path.join(output_dir, 'aic_bic_histograms.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


# ==========================================
# PLOT 3: Renderizado HTML 3D
# ==========================================
def export_interactive_3d_network(coords_3d, p_values, channel_names, filename="plots/red_3d.html", dropped_channels=None, hide_isolated=False):
    """
    Exporta un grafo 3D interactivo en HTML de la conectividad significativa.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    channel_names = list(channel_names)
    coords_3d = np.asarray(coords_3d)
    p_values = np.asarray(p_values)

    keep_mask = np.ones(len(channel_names), dtype=bool)
    if dropped_channels is not None:
        dropped_set = set(dropped_channels)
        keep_mask &= np.array([ch not in dropped_set for ch in channel_names], dtype=bool)
        
    p_threshold = 0.05
    highly_sig_threshold = 0.01

    if hide_isolated:
        sig_mask = (p_values < p_threshold) & ~np.isnan(p_values)
        np.fill_diagonal(sig_mask, False)
        active_nodes = sig_mask.any(axis=0) | sig_mask.any(axis=1)
        keep_mask &= active_nodes

    keep_idx = np.where(keep_mask)[0]
    if keep_idx.size == 0:
        return

    coords_3d = coords_3d[keep_idx]
    p_values = p_values[np.ix_(keep_idx, keep_idx)]
    channel_names = [channel_names[i] for i in keep_idx]
    n_ch = len(channel_names)
    
    xs, ys, zs = coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]
    
    nodos_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers+text',
        marker=dict(size=6, color='black', opacity=0.7),
        text=channel_names,
        textposition="top center",
        hoverinfo='text',
        name='Electrodos'
    )
    
    edge_traces = []
    for i in range(n_ch):
        for j in range(n_ch):
            if i != j and not np.isnan(p_values[i, j]) and p_values[i, j] < p_threshold:
                pval = p_values[i, j]
                
                if pval < highly_sig_threshold:
                    color = 'darkred'
                    width = 4
                else:
                    color = 'red'
                    width = 2
                    
                edge_trace = go.Scatter3d(
                    x=[xs[i], xs[j], None],
                    y=[ys[i], ys[j], None],
                    z=[zs[i], zs[j], None],
                    mode='lines',
                    line=dict(color=color, width=width),
                    hoverinfo='text',
                    text=[f"{channel_names[i]} -> {channel_names[j]} (p={pval:.4f})"],
                    name='Conexión'
                )
                edge_traces.append(edge_trace)

    fig = go.Figure(data=[nodos_trace] + edge_traces)
    fig.update_layout(
        title="Red de Conectividad Significativa (3D)",
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title='')
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    pyo.plot(fig, filename=filename, auto_open=False)


# ==========================================
# PLOT 4: Mapas de Promedio Temporal TFCE
# ==========================================
def plot_tfce_heatmaps(tfce_temporal_avg, band_name, R_label, output_dir="plots"):
    """
    Proyecta las energías TFCE promediadas temporalmente en un heatmap 2D (escala log).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Escala logarítmica para suavizar los picos de energía
    log_values = np.log1p(tfce_temporal_avg)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(
        log_values,
        aspect='auto',
        cmap='inferno',
    )
    plt.colorbar(label='log(1 + TFCE)')
    plt.title(f"Promedio Temporal TFCE - Banda {band_name} (R={R_label})")
    plt.xlabel('Canal Origen')
    plt.ylabel('Canal Destino')
    
    filename = os.path.join(output_dir, f'tfce_heatmap_R{R_label}_{band_name}.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
