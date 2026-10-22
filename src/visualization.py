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
# PLOT 1: Curvas de Selección de Orden MVAR
# ==========================================
def plot_order_selection_curves(lags, aic_means, bic_means, aic_votes, bic_votes, output_dir="plots", suffix="", method_label=None):
    """
    Genera un gráfico formal de 2 paneles para selección de orden MVAR:
    - Panel Izquierdo: Curvas promedio de AIC y BIC vs lag (p) con los mínimos destacados.
    - Panel Derecho: Histogramas de votos por registro (sujeto x condición) para AIC y BIC.
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

    # Panel 2: Distribución de Votos por Registro
    bins = np.arange(lags[0] - 0.5, lags[-1] + 1.5, 1)
    axes[1].hist([aic_votes, bic_votes], bins=bins, label=['Votos AIC', 'Votos BIC'],
                 color=['#1f77b4', '#2ca02c'], edgecolor='black', alpha=0.85)
    axes[1].set_title(f'Orden Óptimo Elegido por Registro{method_str}', fontsize=12)
    axes[1].set_xlabel('Orden Óptimo (p)', fontsize=11)
    axes[1].set_ylabel('Frecuencia (Registros)', fontsize=11)
    axes[1].set_xticks(lags)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    filename = os.path.join(output_dir, f'mvar_order_selection_curves{suffix}.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


# ==========================================
# PLOT 3: Renderizado HTML 3D
# ==========================================
def export_interactive_3d_network(coords_3d, p_values, channel_names, filename="plots/red_3d.html", dropped_channels=None, hide_isolated=False, title="Red de Conectividad Significativa (3D)"):
    """
    Exporta un grafo 3D interactivo en HTML de la conectividad significativa.
    p_values se indexa como [dest, src].
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
                    text=[f"{channel_names[j]} -> {channel_names[i]} (p={pval:.4f})"],
                    name='Conexión'
                )
                edge_traces.append(edge_trace)

    fig = go.Figure(data=[nodos_trace] + edge_traces)
    fig.update_layout(
        title=title,
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
# PLOT 3: Mapas de Contraste por Banda
# ==========================================
def plot_contrast_heatmaps(t_map, p_map, node_names, band_names, title, filename, q_map=None):
    """
    Heatmap del estadístico t (con signo) por banda, matriz [dest, src].
    Marca con '*' las aristas con p < 0.05 sin corregir y con '**' las que
    además sobreviven a la corrección (q_map < 0.05, si se entrega).
    
    t_map, p_map, q_map: np.ndarray (n_dest, n_src, n_bands)
    """
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    n_bands = len(band_names)
    vmax = max(np.nanmax(np.abs(t_map)), 2.0)
    
    fig, axes = plt.subplots(1, n_bands, figsize=(7.5 * n_bands, 6.5), squeeze=False)
    ticks = np.arange(len(node_names))
    for b_idx, b_name in enumerate(band_names):
        ax = axes[0, b_idx]
        t_b = t_map[:, :, b_idx].copy()
        np.fill_diagonal(t_b, np.nan)
        im = ax.imshow(t_b, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        
        for i in range(len(node_names)):
            for j in range(len(node_names)):
                if i == j or np.isnan(p_map[i, j, b_idx]) or p_map[i, j, b_idx] >= 0.05:
                    continue
                mark = '**' if q_map is not None and q_map[i, j, b_idx] < 0.05 else '*'
                ax.text(j, i, mark, ha='center', va='center', fontsize=12, color='black')
                
        ax.set_title(f'Banda {b_name}', fontsize=12)
        ax.set_xticks(ticks)
        ax.set_xticklabels(node_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(ticks)
        ax.set_yticklabels(node_names, fontsize=8)
        ax.set_xlabel('Super-Nodo Origen')
        ax.set_ylabel('Super-Nodo Destino')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='t')
        
    fig.suptitle(f'{title}\n* p < 0.05 sin corregir   ** q < 0.05 (FDR sobre todas las aristas y bandas)', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()


# ==========================================
# PLOT 4: Gráfico de Interacción Grupo x Condición
# ==========================================
def plot_group_condition_means(values_hb, values_si, groups, band_names, filename, title):
    """
    Gráfico de interacción: media ± error estándar por grupo en cada condición,
    con las trayectorias individuales (silencio -> heartbeat) de fondo.
    Líneas paralelas = sin interacción; desplazamiento vertical = efecto de grupo;
    pendiente = efecto de condición.
    
    values_hb, values_si: np.ndarray (n_subj, n_bands)
    groups: np.ndarray (n_subj,) con etiquetas 'FT' / 'PT'
    """
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    colors = {'FT': '#1f77b4', 'PT': '#d62728'}
    offsets = {'FT': -0.04, 'PT': 0.04}
    x = np.array([0, 1])
    
    fig, axes = plt.subplots(1, len(band_names), figsize=(5.5 * len(band_names), 5.5), squeeze=False)
    for b_idx, b_name in enumerate(band_names):
        ax = axes[0, b_idx]
        for grp in ('FT', 'PT'):
            mask = groups == grp
            pairs = np.column_stack([values_si[mask, b_idx], values_hb[mask, b_idx]])
            for row in pairs:
                ax.plot(x + offsets[grp], row, color=colors[grp], alpha=0.12, lw=0.8)
            mean = pairs.mean(axis=0)
            sem = pairs.std(axis=0, ddof=1) / np.sqrt(mask.sum())
            ax.errorbar(x + offsets[grp], mean, yerr=sem, color=colors[grp], marker='o', lw=2.5,
                        capsize=5, label=f'{grp} (n={mask.sum()})')
        ax.set_xticks(x)
        ax.set_xticklabels(['Silencio', 'Heartbeat'])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(f'Banda {b_name}', fontsize=12)
        ax.set_ylabel('dDTF medio')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)
        
    fig.suptitle(f'{title}\nMedia ± EE; líneas tenues = sujetos individuales', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()


# ==========================================
# PLOT 5: Comparación de Robustez (Mean vs PCA)
# ==========================================
def plot_robustness_comparison(t_file_mean="plots/t_values_group_mean.npy",
                               t_file_pca="plots/t_values_group_pca.npy",
                               band_names=None, output_dir="plots"):
    """
    Dispersión de los estadísticos t del efecto principal de grupo (FT vs PT) obtenidos
    con agregación 'mean' contra los obtenidos con 'pca', arista por arista y por banda.
    Retorna la ruta del gráfico y las correlaciones de Pearson por banda.
    """
    if band_names is None:
        band_names = list(parameters.F_BANDS.keys())
        
    for path in (t_file_mean, t_file_pca):
        if not os.path.exists(path):
            print(f"[AVISO] No se encontró {path} para la comparación de robustez.")
            return None, None
            
    t_mean = np.load(t_file_mean)  # (dest, src, band)
    t_pca = np.load(t_file_pca)
    off_diag = ~np.eye(t_mean.shape[0], dtype=bool)
    
    fig, axes = plt.subplots(1, len(band_names), figsize=(6 * len(band_names), 5.5), squeeze=False)
    correlations = {}
    for b_idx, b_name in enumerate(band_names):
        x = t_mean[:, :, b_idx][off_diag]
        y = t_pca[:, :, b_idx][off_diag]
        r = np.corrcoef(x, y)[0, 1]
        correlations[b_name] = r
        
        ax = axes[0, b_idx]
        lim = max(np.abs(x).max(), np.abs(y).max(), 2.5) * 1.05
        ax.scatter(x, y, s=18, alpha=0.7, color='#1f77b4')
        ax.plot([-lim, lim], [-lim, lim], color='gray', lw=1, linestyle='--')
        for v in (-1.99, 1.99):
            ax.axvline(v, color='red', lw=0.8, linestyle=':')
            ax.axhline(v, color='red', lw=0.8, linestyle=':')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f'Banda {b_name}: r = {r:.2f}', fontsize=12)
        ax.set_xlabel('t FT vs PT (ROI: Mean)')
        ax.set_ylabel('t FT vs PT (ROI: PCA)')
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "robustness_comparison_mean_vs_pca.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"[ROBUSTEZ] Gráfico comparativo generado: {out_path}")
    return out_path, correlations
