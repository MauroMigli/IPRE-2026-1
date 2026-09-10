import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
except ImportError:
    go = None
    pyo = None
import parameters

# ==========================================
# PLOT 1: Conteo de Aristas (Naive vs E[FP])
# ==========================================
def plot_edge_counts(epochs_x, naive_counts, fdr_counts, tfce_counts, expected_fp, band_name, output_dir="plots", suffix="", method_label=None):
    """
    Genera el gráfico de líneas mostrando las aristas que sobreviven a Naive, FDR y TFCE,
    comparado contra la esperanza matemática de falsos positivos (E[FP]).
    Permite sufijos (e.g. '_mean', '_pca') para evitar sobreescritura en análisis de robustez.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs_x, naive_counts, marker='o', label='Naive (p < 0.05)', color='gray', alpha=0.7)
    plt.plot(epochs_x, fdr_counts, marker='s', label='FDR (q < 0.05)', color='blue')
    if tfce_counts is not None:
        plt.plot(epochs_x, tfce_counts, marker='^', label='TFCE (p < 0.05)', color='green')
    
    # Línea teórica de Falsos Positivos
    plt.axhline(y=expected_fp, color='red', linestyle='--', label=f'Esperanza FP (E[FP] = {expected_fp:.1f})')
    
    method_str = f" [{method_label}]" if method_label else ""
    plt.title(f'Evolución temporal de aristas significativas - Banda {band_name}{method_str}')
    plt.xlabel('Época (Tiempo)')
    plt.ylabel('Cantidad de Aristas Significativas')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    filename = os.path.join(output_dir, f'edge_counts_{band_name}{suffix}.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    # Si tiene sufijo específico pero es mean, guardar también la copia base
    if suffix == "_mean":
        plt.savefig(os.path.join(output_dir, f'edge_counts_{band_name}.png'), bbox_inches='tight', dpi=300)
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


def plot_order_selection_curves(lags, aic_means, bic_means, aic_votes, bic_votes, output_dir="plots", suffix="", method_label=None):
    """
    Genera un gráfico formal de 2 paneles para selección de orden MVAR:
    - Panel Izquierdo: Curvas promedio de AIC y BIC vs lag (p) con los mínimos destacados.
    - Panel Derecho: Histogramas de votos por época para AIC y BIC.
    Permite sufijos (e.g. '_mean', '_pca') para comparar criterios bajo distintas agregaciones.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    method_str = f" [{method_label}]" if method_label else ""

    # Panel 1: Curvas de Información Promedio
    axes[0].plot(lags, aic_means, marker='o', label='AIC', color='#1f77b4', lw=2)
    axes[0].plot(lags, bic_means, marker='s', label='BIC', color='#2ca02c', lw=2)

    best_p_aic = lags[np.nanargmin(aic_means)]
    best_p_bic = lags[np.nanargmin(bic_means)]

    axes[0].axvline(x=best_p_aic, color='#1f77b4', linestyle='--', alpha=0.7, label=f'Mínimo AIC (p={best_p_aic})')
    axes[0].axvline(x=best_p_bic, color='#2ca02c', linestyle='--', alpha=0.7, label=f'Mínimo BIC (p={best_p_bic})')

    axes[0].set_title(f'Criterios de Información Promedio vs. Rezago MVAR (p){method_str}', fontsize=12)
    axes[0].set_xlabel('Orden del Modelo (p)', fontsize=11)
    axes[0].set_ylabel('Criterio de Información', fontsize=11)
    axes[0].set_xticks(lags)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=10)

    # Panel 2: Distribución de Votos por Época
    bins = np.arange(lags[0] - 0.5, lags[-1] + 1.5, 1)
    axes[1].hist([aic_votes, bic_votes], bins=bins, label=['Votos AIC', 'Votos BIC'],
                 color=['#1f77b4', '#2ca02c'], edgecolor='black', alpha=0.85)
    axes[1].set_title(f'Distribución de Orden Óptimo Elegido por Época{method_str}', fontsize=12)
    axes[1].set_xlabel('Orden Óptimo (p)', fontsize=11)
    axes[1].set_ylabel('Frecuencia (Épocas)', fontsize=11)
    axes[1].set_xticks(lags)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    filename = os.path.join(output_dir, f'mvar_order_selection_curves{suffix}.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    if suffix == "_mean":
        plt.savefig(os.path.join(output_dir, 'mvar_order_selection_curves.png'), bbox_inches='tight', dpi=300)
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
        if any(ch in dropped_set for ch in channel_names):
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
    marker_size = 10 if any('_' in ch for ch in channel_names) else 6
    
    nodos_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers+text',
        marker=dict(size=marker_size, color='black', opacity=0.7),
        text=channel_names,
        textposition="top center",
        hoverinfo='text',
        name='Nodos'
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
def plot_tfce_heatmaps(tfce_temporal_avg, band_name, R_label, output_dir="plots", node_names=None):
    """
    Proyecta las energías TFCE promediadas temporalmente en un heatmap 2D (escala log).
    Soporta etiquetas de super-nodos o canales.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    log_values = np.log1p(tfce_temporal_avg)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(
        log_values,
        aspect='auto',
        cmap='inferno',
    )
    plt.colorbar(label='log(1 + TFCE)')
    plt.title(f"Promedio Temporal TFCE - Banda {band_name} (R={R_label})")
    
    if node_names is not None:
        ticks = np.arange(len(node_names))
        plt.xticks(ticks, node_names, rotation=45, ha='right', fontsize=9)
        plt.yticks(ticks, node_names, fontsize=9)
        plt.xlabel('Super-Nodo Origen')
        plt.ylabel('Super-Nodo Destino')
    else:
        plt.xlabel('Canal Origen')
        plt.ylabel('Canal Destino')
        
    plt.tight_layout()
    filename = os.path.join(output_dir, f'tfce_heatmap_R{R_label}_{band_name}.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


# ==============================================================================
# PLOT 5: Red 3D Temporal Interactiva con Slider de Épocas y Animación
# ==============================================================================
def export_interactive_temporal_3d_network(
    coords_3d,
    p_values_band,
    node_names,
    band_name,
    filename="plots/red_temporal_3d.html",
    p_threshold=0.05,
    epoch_duration=0.5
):
    """
    Exporta un grafo 3D interactivo en HTML con un slider temporal y botones de reproducción
    para observar la evolución de las conexiones significativas a través de las épocas.
    
    Parámetros:
    -----------
    coords_3d: (n_nodes, 3) coordenadas de los nodos
    p_values_band: (n_dest, n_src, n_epochs) matriz de p-valores para una banda
    node_names: lista de nombres de los nodos (ROIs)
    band_name: nombre de la banda (ej: 'Gamma', 'Delta')
    filename: ruta del archivo HTML de salida
    p_threshold: umbral de significancia (default: 0.05)
    epoch_duration: duración de cada época en segundos (default: 0.5s)
    """
    if go is None or pyo is None:
        raise ImportError("plotly es requerido para exportar gráficos 3D interactivos.")
        
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    node_names = list(node_names)
    coords_3d = np.asarray(coords_3d)
    p_values_band = np.asarray(p_values_band)
    
    n_nodes = len(node_names)
    n_epochs = p_values_band.shape[2]
    
    xs, ys, zs = coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]
    
    # Traza 0: Nodos cerebrales
    nodes_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers+text',
        marker=dict(size=10, color='#2c3e50', opacity=0.85, line=dict(color='white', width=1)),
        text=node_names,
        textposition="top center",
        hoverinfo='text',
        name='Super-Nodos'
    )
    
    all_traces = [nodes_trace]
    
    sig_counts_per_ep = []
    for e in range(n_epochs):
        p_ep = p_values_band[:, :, e].copy()
        np.fill_diagonal(p_ep, np.nan)
        sig_counts_per_ep.append(int(np.nansum(p_ep < p_threshold)))
        
    default_ep = int(np.argmax(sig_counts_per_ep))
    
    for e in range(n_epochs):
        p_ep = p_values_band[:, :, e]
        edge_x, edge_y, edge_z = [], [], []
        edge_hover = []
        
        for dest in range(n_nodes):
            for src in range(n_nodes):
                if dest != src and not np.isnan(p_ep[dest, src]) and p_ep[dest, src] < p_threshold:
                    pval = p_ep[dest, src]
                    edge_x.extend([xs[src], xs[dest], None])
                    edge_y.extend([ys[src], ys[dest], None])
                    edge_z.extend([zs[src], zs[dest], None])
                    hover_txt = f"{node_names[src]} → {node_names[dest]} (p = {pval:.4f})"
                    edge_hover.extend([hover_txt, hover_txt, None])
                    
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='red', width=3.5),
            hoverinfo='text',
            text=edge_hover,
            name=f'Época {e}',
            visible=(e == default_ep)
        )
        all_traces.append(edge_trace)
        
    steps = []
    for e in range(n_epochs):
        visibility = [True] + [(idx == e) for idx in range(n_epochs)]
        t_start = e * epoch_duration
        t_end = (e + 1) * epoch_duration
        count = sig_counts_per_ep[e]
        
        title_text = (
            f"Conectividad Temporal (3D) - Banda {band_name}<br>"
            f"<sup>Época {e} ({t_start:.1f}s - {t_end:.1f}s) | "
            f"<b>{count}</b> conexiones significativas (p < {p_threshold})</sup>"
        )
        
        step = dict(
            method="update",
            label=f"Ep {e}",
            args=[
                {"visible": visibility},
                {"title.text": title_text}
            ]
        )
        steps.append(step)
        
    sliders = [dict(
        active=default_ep,
        currentvalue=dict(prefix="Línea de Tiempo: ", visible=True, xanchor="center", font=dict(size=12)),
        pad=dict(t=50, b=10),
        len=0.9,
        x=0.05,
        y=0.02,
        steps=steps
    )]
    
    updatemenus = [dict(
        type="buttons",
        showactive=False,
        x=0.05,
        y=0.12,
        xanchor="left",
        yanchor="top",
        pad=dict(t=10, r=10),
        buttons=[
            dict(
                label="▶ Reproducir",
                method="animate",
                args=[None, dict(frame=dict(duration=800, redraw=True), fromcurrent=True)]
            ),
            dict(
                label="⏸ Pausar",
                method="animate",
                args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]
            )
        ]
    )]
    
    frames = []
    for e in range(n_epochs):
        vis = [True] + [(idx == e) for idx in range(n_epochs)]
        t_start = e * epoch_duration
        t_end = (e + 1) * epoch_duration
        count = sig_counts_per_ep[e]
        
        title_text = (
            f"Conectividad Temporal (3D) - Banda {band_name}<br>"
            f"<sup>Época {e} ({t_start:.1f}s - {t_end:.1f}s) | "
            f"<b>{count}</b> conexiones significativas (p < {p_threshold})</sup>"
        )
        
        frame = go.Frame(
            data=[all_traces[0]] + [
                go.Scatter3d(visible=(idx == e)) for idx in range(n_epochs)
            ],
            name=f"Ep {e}",
            layout=dict(title_text=title_text)
        )
        frames.append(frame)
        
    initial_title = (
        f"Conectividad Temporal (3D) - Banda {band_name}<br>"
        f"<sup>Época {default_ep} ({default_ep*epoch_duration:.1f}s - {(default_ep+1)*epoch_duration:.1f}s) | "
        f"<b>{sig_counts_per_ep[default_ep]}</b> conexiones significativas (p < {p_threshold})</sup>"
    )
    
    fig = go.Figure(
        data=all_traces,
        layout=go.Layout(
            title=dict(text=initial_title, x=0.5, font=dict(size=15)),
            showlegend=False,
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title=''),
                yaxis=dict(showbackground=False, showticklabels=False, title=''),
                zaxis=dict(showbackground=False, showticklabels=False, title='')
            ),
            margin=dict(l=0, r=0, b=100, t=60),
            sliders=sliders,
            updatemenus=updatemenus
        ),
        frames=frames
    )
    
    pyo.plot(fig, filename=filename, auto_open=False)
    print(f"  [OK] Grafo temporal 3D exportado a: {filename}")


# ==========================================
# PLOT 5: Comparación de Robustez (Mean vs PCA)
# ==========================================
def plot_robustness_comparison(p_file_mean="plots/p_values_rois_mean_naive.npy", 
                               p_file_pca="plots/p_values_rois_pca_naive.npy", 
                               band_names=None, output_dir="plots"):
    """
    Genera un gráfico comparativo de robustez metodológica:
    Contrasta la evolución temporal de aristas significativas obtenidas con
    agregación espacial 'mean' vs. primer componente principal 'pca'.
    """
    if band_names is None:
        band_names = list(parameters.F_BANDS.keys())
        
    if not os.path.exists(p_file_mean):
        fallback = "plots/p_values_rois_naive.npy"
        if os.path.exists(fallback):
            p_file_mean = fallback
        else:
            print(f"[AVISO] No se encontró {p_file_mean} para la comparación de robustez.")
            return None
            
    if not os.path.exists(p_file_pca):
        print(f"[AVISO] No se encontró {p_file_pca} para la comparación de robustez.")
        return None
        
    p_mean = np.load(p_file_mean)  # (dest, src, band, epoch)
    p_pca = np.load(p_file_pca)    # (dest, src, band, epoch)
    
    n_epochs = min(p_mean.shape[3], p_pca.shape[3])
    epochs_x = np.arange(n_epochs)
    n_nodes = p_mean.shape[0]
    expected_fp = (n_nodes * (n_nodes - 1)) * 0.05
    
    target_bands = ["Delta", "Gamma"]  # Bandas biológicamente relevantes
    fig, axes = plt.subplots(1, len(target_bands), figsize=(14, 5), sharey=True)
    if len(target_bands) == 1:
        axes = [axes]
        
    for idx, b_name in enumerate(target_bands):
        b_idx = band_names.index(b_name)
        
        counts_mean = [np.nansum(p_mean[:, :, b_idx, e] < 0.05) for e in epochs_x]
        counts_pca = [np.nansum(p_pca[:, :, b_idx, e] < 0.05) for e in epochs_x]
        
        axes[idx].plot(epochs_x, counts_mean, marker='o', lw=2, color='#1f77b4', label='ROI: Promedio Espacial (Mean)')
        axes[idx].plot(epochs_x, counts_pca, marker='s', lw=2, linestyle='--', color='#ff7f0e', label='ROI: 1er Comp. PCA (PC1)')
        axes[idx].axhline(y=expected_fp, color='red', linestyle=':', label=f'Esperanza FP ({expected_fp:.1f})')
        
        axes[idx].set_title(f'Banda {b_name}: Comparación de Robustez', fontsize=12)
        axes[idx].set_xlabel('Época (Tiempo)', fontsize=11)
        if idx == 0:
            axes[idx].set_ylabel('Aristas Significativas (p < 0.05)', fontsize=11)
        axes[idx].grid(True, linestyle='--', alpha=0.6)
        axes[idx].legend(fontsize=9)
        axes[idx].set_xticks(epochs_x)
        
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "robustness_comparison_mean_vs_pca.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[ROBUSTEZ] Gráfico comparativo generado: {out_path}")
    return out_path

