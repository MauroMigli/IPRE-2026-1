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
def export_interactive_3d_network(coords_3d, p_values, channel_names, filename="plots/red_3d.html", dropped_channels=None, hide_isolated=False, t_values=None):
    """
    Exporta un grafo 3D interactivo en HTML de la conectividad significativa.
    Si se suministra t_values, colorea las aristas según la dirección del efecto:
      - Rojo: t > 0 (FT > PT, mayor conectividad en recién nacidos a término)
      - Azul: t < 0 (PT > FT, mayor conectividad en prematuros)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    channel_names = list(channel_names)
    coords_3d = np.asarray(coords_3d)
    p_values = np.asarray(p_values)
    t_mat = np.asarray(t_values) if t_values is not None else None

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
    if t_mat is not None:
        t_mat = t_mat[np.ix_(keep_idx, keep_idx)]
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
    for i in range(n_ch):       # dest
        for j in range(n_ch):   # src
            if i != j and not np.isnan(p_values[i, j]) and p_values[i, j] < p_threshold:
                pval = p_values[i, j]
                
                if t_mat is not None and not np.isnan(t_mat[i, j]):
                    t_val = t_mat[i, j]
                    is_pos = (t_val > 0)
                    dir_str = "FT > PT" if is_pos else "PT > FT"
                    t_str = f", t = {t_val:+.2f} ({dir_str})"
                    color = ('#b71c1c' if pval < highly_sig_threshold else '#e74c3c') if is_pos else ('#0d47a1' if pval < highly_sig_threshold else '#1f77b4')
                else:
                    t_str = ""
                    color = 'darkred' if pval < highly_sig_threshold else 'red'
                    
                width = 4 if pval < highly_sig_threshold else 2.5
                    
                edge_trace = go.Scatter3d(
                    x=[xs[j], xs[i], None],
                    y=[ys[j], ys[i], None],
                    z=[zs[j], zs[i], None],
                    mode='lines',
                    line=dict(color=color, width=width),
                    hoverinfo='text',
                    text=[f"{channel_names[j]} → {channel_names[i]} (p={pval:.4f}{t_str})"],
                    name='Conexión'
                )
                edge_traces.append(edge_trace)

    fig = go.Figure(data=[nodos_trace] + edge_traces)
    
    title_text = "Red de Conectividad Significativa (3D)"
    if t_mat is not None:
        title_text += (
            "<br><sup><span style='color:#e74c3c;font-weight:bold;'>■ FT > PT (t > 0)</span>"
            " &nbsp;&nbsp;&nbsp;&nbsp; "
            "<span style='color:#1f77b4;font-weight:bold;'>■ PT > FT (t < 0)</span></sup>"
        )
        
    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=14)),
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title='')
        ),
        margin=dict(l=0, r=0, b=0, t=50)
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
    epoch_duration=0.5,
    t_values_band=None
):
    """
    Exporta un grafo 3D interactivo en HTML con un slider temporal y botones de reproducción
    para observar la evolución de las conexiones significativas a través de las épocas.
    Si se suministra t_values_band, colorea las aristas según el signo de t:
      - Rojo: t > 0 (FT > PT, mayor reactividad en a término)
      - Azul: t < 0 (PT > FT, mayor reactividad en prematuros)
    
    Parámetros:
    -----------
    coords_3d: (n_nodes, 3) coordenadas de los nodos
    p_values_band: (n_dest, n_src, n_epochs) matriz de p-valores para una banda
    node_names: lista de nombres de los nodos (ROIs)
    band_name: nombre de la banda (ej: 'Gamma', 'Delta')
    filename: ruta del archivo HTML de salida
    p_threshold: umbral de significancia (default: 0.05)
    epoch_duration: duración de cada época en segundos (default: 0.5s)
    t_values_band: (n_dest, n_src, n_epochs) opcional matriz de estadísticos t
    """
    if go is None or pyo is None:
        raise ImportError("plotly es requerido para exportar gráficos 3D interactivos.")
        
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    node_names = list(node_names)
    coords_3d = np.asarray(coords_3d)
    p_values_band = np.asarray(p_values_band)
    has_t = (t_values_band is not None)
    t_band = np.asarray(t_values_band) if has_t else None
    
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
    sig_counts_pos = []
    sig_counts_neg = []
    
    for e in range(n_epochs):
        p_ep = p_values_band[:, :, e].copy()
        np.fill_diagonal(p_ep, np.nan)
        mask_sig = (p_ep < p_threshold) & ~np.isnan(p_ep)
        sig_counts_per_ep.append(int(np.sum(mask_sig)))
        
        if has_t:
            t_ep = t_band[:, :, e]
            sig_counts_pos.append(int(np.sum(mask_sig & (t_ep > 0))))
            sig_counts_neg.append(int(np.sum(mask_sig & (t_ep < 0))))
        else:
            sig_counts_pos.append(int(np.sum(mask_sig)))
            sig_counts_neg.append(0)
        
    default_ep = int(np.argmax(sig_counts_per_ep))
    
    for e in range(n_epochs):
        p_ep = p_values_band[:, :, e]
        t_ep = t_band[:, :, e] if has_t else None
        
        if has_t:
            pos_x, pos_y, pos_z, pos_hover = [], [], [], []
            neg_x, neg_y, neg_z, neg_hover = [], [], [], []
            
            for dest in range(n_nodes):
                for src in range(n_nodes):
                    if dest != src and not np.isnan(p_ep[dest, src]) and p_ep[dest, src] < p_threshold:
                        pval = p_ep[dest, src]
                        tval = t_ep[dest, src]
                        
                        if tval > 0:
                            pos_x.extend([xs[src], xs[dest], None])
                            pos_y.extend([ys[src], ys[dest], None])
                            pos_z.extend([zs[src], zs[dest], None])
                            hover_txt = f"{node_names[src]} → {node_names[dest]} (t = {tval:+.2f}, p = {pval:.4f}) [FT > PT]"
                            pos_hover.extend([hover_txt, hover_txt, None])
                        else:
                            neg_x.extend([xs[src], xs[dest], None])
                            neg_y.extend([ys[src], ys[dest], None])
                            neg_z.extend([zs[src], zs[dest], None])
                            hover_txt = f"{node_names[src]} → {node_names[dest]} (t = {tval:+.2f}, p = {pval:.4f}) [PT > FT]"
                            neg_hover.extend([hover_txt, hover_txt, None])
                            
            trace_pos = go.Scatter3d(
                x=pos_x, y=pos_y, z=pos_z,
                mode='lines',
                line=dict(color='#e74c3c', width=3.5),
                hoverinfo='text',
                text=pos_hover,
                name=f'FT > PT (Ep {e})',
                visible=(e == default_ep)
            )
            trace_neg = go.Scatter3d(
                x=neg_x, y=neg_y, z=neg_z,
                mode='lines',
                line=dict(color='#1f77b4', width=3.5),
                hoverinfo='text',
                text=neg_hover,
                name=f'PT > FT (Ep {e})',
                visible=(e == default_ep)
            )
            all_traces.extend([trace_pos, trace_neg])
        else:
            edge_x, edge_y, edge_z, edge_hover = [], [], [], []
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
    traces_per_epoch = 2 if has_t else 1
    
    for e in range(n_epochs):
        visibility = [True] + [((idx // traces_per_epoch) == e) for idx in range(n_epochs * traces_per_epoch)]
        t_start = e * epoch_duration
        t_end = (e + 1) * epoch_duration
        count = sig_counts_per_ep[e]
        
        if has_t:
            count_pos = sig_counts_pos[e]
            count_neg = sig_counts_neg[e]
            title_text = (
                f"Conectividad Temporal (3D) - Banda {band_name}<br>"
                f"<sup>Época {e} ({t_start:.1f}s - {t_end:.1f}s) | "
                f"<b>{count}</b> conexiones (p < {p_threshold}) : "
                f"<span style='color:#e74c3c;font-weight:bold;'>{count_pos} FT > PT</span> | "
                f"<span style='color:#1f77b4;font-weight:bold;'>{count_neg} PT > FT</span></sup>"
            )
        else:
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
        t_start = e * epoch_duration
        t_end = (e + 1) * epoch_duration
        count = sig_counts_per_ep[e]
        
        if has_t:
            count_pos = sig_counts_pos[e]
            count_neg = sig_counts_neg[e]
            title_text = (
                f"Conectividad Temporal (3D) - Banda {band_name}<br>"
                f"<sup>Época {e} ({t_start:.1f}s - {t_end:.1f}s) | "
                f"<b>{count}</b> conexiones (p < {p_threshold}) : "
                f"<span style='color:#e74c3c;font-weight:bold;'>{count_pos} FT > PT</span> | "
                f"<span style='color:#1f77b4;font-weight:bold;'>{count_neg} PT > FT</span></sup>"
            )
        else:
            title_text = (
                f"Conectividad Temporal (3D) - Banda {band_name}<br>"
                f"<sup>Época {e} ({t_start:.1f}s - {t_end:.1f}s) | "
                f"<b>{count}</b> conexiones significativas (p < {p_threshold})</sup>"
            )
        
        frame_traces = [all_traces[0]]
        for k in range(n_epochs * traces_per_epoch):
            frame_traces.append(go.Scatter3d(visible=((k // traces_per_epoch) == e)))
            
        frame = go.Frame(
            data=frame_traces,
            name=f"Ep {e}",
            layout=dict(title_text=title_text)
        )
        frames.append(frame)
        
    t_start_def = default_ep * epoch_duration
    t_end_def = (default_ep + 1) * epoch_duration
    if has_t:
        initial_title = (
            f"Conectividad Temporal (3D) - Banda {band_name}<br>"
            f"<sup>Época {default_ep} ({t_start_def:.1f}s - {t_end_def:.1f}s) | "
            f"<b>{sig_counts_per_ep[default_ep]}</b> conexiones (p < {p_threshold}) : "
            f"<span style='color:#e74c3c;font-weight:bold;'>{sig_counts_pos[default_ep]} FT > PT</span> | "
            f"<span style='color:#1f77b4;font-weight:bold;'>{sig_counts_neg[default_ep]} PT > FT</span></sup>"
        )
    else:
        initial_title = (
            f"Conectividad Temporal (3D) - Banda {band_name}<br>"
            f"<sup>Época {default_ep} ({t_start_def:.1f}s - {t_end_def:.1f}s) | "
            f"<b>{sig_counts_per_ep[default_ep]}</b> conexiones significativas (p < {p_threshold})</sup>"
        )
    
    annotations = [
        dict(
            text="<span style='color:#e74c3c;font-weight:bold;'>■ FT > PT (t > 0)</span> &nbsp;&nbsp;&nbsp;&nbsp; <span style='color:#1f77b4;font-weight:bold;'>■ PT > FT (t < 0)</span>",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.98,
            xanchor="center", yanchor="top",
            font=dict(size=13)
        )
    ] if has_t else []
    
    fig = go.Figure(
        data=all_traces,
        layout=go.Layout(
            title=dict(text=initial_title, x=0.5, font=dict(size=15)),
            showlegend=False,
            annotations=annotations,
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


# ==============================================================================
# PLOT 6: Evolución Temporal Desglosada por Dirección del Signo t (FT vs PT)
# ==============================================================================
def plot_direction_counts(epochs_x, pos_counts, neg_counts, expected_fp, band_name, output_dir="plots", suffix="", method_label=None):
    """
    Genera el gráfico temporal desglosado por dirección del efecto estadístico:
    - Conexiones con t > 0 (FT > PT, rojo: mayor reactividad en a término)
    - Conexiones con t < 0 (PT > FT, azul: mayor reactividad en prematuros)
    - Total de aristas significativas (línea negra punteada)
    - Esperanza matemática de falsos positivos E[FP] (línea gris discontinua)
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    total_counts = np.array(pos_counts) + np.array(neg_counts)
    
    plt.plot(epochs_x, total_counts, marker='o', color='black', linestyle=':', lw=1.5, alpha=0.6, label='Total Significativas (p < 0.05)')
    plt.plot(epochs_x, pos_counts, marker='^', color='#d62728', lw=2.2, label='FT > PT (t > 0, mayor respuesta en Término)')
    plt.plot(epochs_x, neg_counts, marker='v', color='#1f77b4', lw=2.2, label='PT > FT (t < 0, mayor respuesta en Prematuro)')
    
    # Línea teórica de Falsos Positivos
    plt.axhline(y=expected_fp, color='gray', linestyle='--', alpha=0.7, label=f'Esperanza FP (E[FP] = {expected_fp:.1f})')
    
    method_str = f" [{method_label}]" if method_label else ""
    plt.title(f'Dirección de Conectividad Significativa (Signo t) - Banda {band_name}{method_str}', fontsize=12)
    plt.xlabel('Época (Tiempo)', fontsize=11)
    plt.ylabel('Cantidad de Conexiones', fontsize=11)
    plt.xticks(epochs_x)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    filename = os.path.join(output_dir, f'edge_counts_direction_{band_name}{suffix}.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    if suffix == "_mean":
        plt.savefig(os.path.join(output_dir, f'edge_counts_direction_{band_name}.png'), bbox_inches='tight', dpi=300)
    plt.close()


# ==============================================================================
# PLOT 7: Matriz Direccional t de Welch con Mapa Divergente y Significatividad
# ==============================================================================
def plot_t_matrix_heatmap(t_matrix, p_matrix, node_names, band_name, epoch_idx, output_dir="plots", suffix="", vmin=None, vmax=None):
    """
    Genera un mapa de calor (heatmap 2D) de la matriz de adyacencia dirigida origen -> destino
    con colormap divergente centrado en 0 (azul para t < 0, rojo para t > 0).
    Anota los valores t e indica significatividad estadística (* si p < 0.05, ** si p < 0.01).
    """
    os.makedirs(output_dir, exist_ok=True)
    n_nodes = len(node_names)
    
    # Excluir diagonal
    t_plot = np.array(t_matrix, dtype=float).copy()
    p_plot = np.array(p_matrix, dtype=float).copy()
    np.fill_diagonal(t_plot, np.nan)
    np.fill_diagonal(p_plot, np.nan)
    
    # Límite simétrico de color
    max_abs = np.nanmax(np.abs(t_plot)) if np.any(~np.isnan(t_plot)) else 3.0
    if max_abs == 0 or np.isnan(max_abs): max_abs = 3.0
    if vmax is None: vmax = max(max_abs, 2.5)
    if vmin is None: vmin = -vmax
    
    fig, ax = plt.subplots(figsize=(9, 8))
    cax = ax.imshow(t_plot, cmap='coolwarm', vmin=vmin, vmax=vmax, aspect='auto')
    
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Estadístico t de Welch (Azul: PT > FT | Rojo: FT > PT)', fontsize=11)
    
    ticks = np.arange(n_nodes)
    ax.set_xticks(ticks)
    ax.set_xticklabels(node_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(ticks)
    ax.set_yticklabels(node_names, fontsize=9)
    
    ax.set_xlabel('Super-Nodo Origen (src)', fontsize=11)
    ax.set_ylabel('Super-Nodo Destino (dest)', fontsize=11)
    ax.set_title(f'Matriz Direccional t de Welch - Banda {band_name} (Época {epoch_idx})\n[* p < 0.05, ** p < 0.01]', fontsize=12)
    
    # Anotación textual por celda
    for i in range(n_nodes):       # dest
        for j in range(n_nodes):   # src
            if i == j or np.isnan(t_plot[i, j]):
                continue
            t_val = t_plot[i, j]
            p_val = p_plot[i, j]
            
            sig_star = ""
            if p_val < 0.01:
                sig_star = "**"
            elif p_val < 0.05:
                sig_star = "*"
                
            text_color = "white" if abs(t_val) > (vmax * 0.55) else "black"
            fontweight = "bold" if sig_star else "normal"
            ax.text(j, i, f"{t_val:+.2f}{sig_star}", ha="center", va="center", color=text_color, fontsize=8, fontweight=fontweight)
            
    plt.tight_layout()
    filename = os.path.join(output_dir, f't_heatmap_{band_name}_e{epoch_idx}{suffix}.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    if suffix == "_mean":
        plt.savefig(os.path.join(output_dir, f't_heatmap_{band_name}_e{epoch_idx}.png'), bbox_inches='tight', dpi=300)
    plt.close()


