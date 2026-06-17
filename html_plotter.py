import plotly.graph_objects as go
import plotly.offline as pyo
import numpy as np

def export_interactive_3d_network(
    coords_3d,
    p_values,
    channel_names,
    filename="red_conectividad_3d.html",
    dropped_channels=None,
    hide_isolated_nodes=True,
):
    print(f"\n--- Generando visualización interactiva en {filename} ---")
    
    # Filtrado defensivo: evita mostrar nodos inválidos o aislados.
    channel_names = list(channel_names)
    coords_3d = np.asarray(coords_3d)
    p_values = np.asarray(p_values)

    keep_mask = np.ones(len(channel_names), dtype=bool)
    if dropped_channels is not None:
        dropped_set = set(dropped_channels)
        keep_mask &= np.array([ch not in dropped_set for ch in channel_names], dtype=bool)

    n_ch_active = np.sum(keep_mask) 
    n_tests = max(n_ch_active * (n_ch_active - 1), 1)
    
    # --- 1. DEFINICIÓN DE UMBRALES ---
    p_threshold = 0.05
    highly_sig_threshold = 0.01
    extremely_sig_threshold = 0.001
    # ---------------------------------

    # Excluir nodos sin ninguna conexión significativa (grafo "sucio").
    if hide_isolated_nodes:
        sig_mask = (p_values < p_threshold) & ~np.isnan(p_values)
        np.fill_diagonal(sig_mask, False)
        active_nodes = sig_mask.any(axis=0) | sig_mask.any(axis=1)
        keep_mask &= active_nodes

    keep_idx = np.where(keep_mask)[0]
    if keep_idx.size == 0:
        print("No hay nodos válidos con conexiones significativas para graficar.")
        return

    if keep_idx.size < len(channel_names):
        removed = [channel_names[i] for i in np.where(~keep_mask)[0]]
        print(f"Nodos excluidos del gráfico: {removed}")

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
    
    edges_traces = []
    cone_x, cone_y, cone_z = [], [], []
    cone_u, cone_v, cone_w = [], [], []
    cone_colors = []
    
    for i in range(n_ch):
        for j in range(n_ch):
            # Ignorar la diagonal y filtrar por significancia
            if i != j and not np.isnan(p_values[i, j]) and p_values[i, j] < p_threshold:
                x_src, y_src, z_src = coords_3d[j]
                x_dest, y_dest, z_dest = coords_3d[i]
                
                # --- 2. LÓGICA DE ESCALA DE COLORES Y GROSOR ---
                if p_values[i, j] < extremely_sig_threshold:
                    color = 'darkred'
                    width = 6
                    color_val = 2 # Nuevo valor para los conos
                elif p_values[i, j] < highly_sig_threshold:
                    color = 'red'
                    width = 4
                    color_val = 1 
                else:
                    color = 'orange'
                    width = 2
                    color_val = 0
                # -----------------------------------------------
                
                # Línea de conexión
                edge_trace = go.Scatter3d(
                    x=[x_src, x_dest],
                    y=[y_src, y_dest],
                    z=[z_src, z_dest],
                    mode='lines',
                    line=dict(color=color, width=width),
                    hoverinfo='none',
                    showlegend=False
                )
                edges_traces.append(edge_trace)

                # Vectores para el cono (Flecha direccional)
                u = x_dest - x_src
                v = y_dest - y_src
                w = z_dest - z_src
                
                cone_x.append(x_src + u * 0.75)
                cone_y.append(y_src + v * 0.75)
                cone_z.append(z_src + w * 0.75)
                cone_u.append(u)
                cone_v.append(v)
                cone_w.append(w)
                cone_colors.append(color_val)

    # --- 3. ACTUALIZACIÓN DE LAS PUNTAS DE FLECHA ---
    if cone_x:
        arrows_trace = go.Cone(
            x=cone_x, y=cone_y, z=cone_z,
            u=cone_u, v=cone_v, w=cone_w,
            sizemode="absolute",
            sizeref=0.5,
            anchor="tip",
            # Actualizamos la escala: 0=naranja, 0.5=rojo, 1=rojo oscuro
            colorscale=[[0, 'orange'], [0.5, 'red'], [1, 'darkred']],
            cmin=0, cmax=2, # Ampliamos el rango máximo a 2
            showscale=False,
            hoverinfo='none'
        )
        edges_traces.append(arrows_trace)

    # --- 4. ACTUALIZACIÓN DEL TÍTULO ---
    layout = go.Layout(
        title=f"Red Significativa<br>Naranja: p < {p_threshold} | Rojo: p < {highly_sig_threshold} | Rojo Oscuro: p < {extremely_sig_threshold}",
        scene=dict(
            xaxis=dict(title='X', showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title='Y', showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(title='Z', showgrid=False, zeroline=False, showticklabels=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig = go.Figure(data=[nodos_trace] + edges_traces, layout=layout)
    pyo.plot(fig, filename=filename, auto_open=False)
    print(f"Archivo generado exitosamente.")

def get_3d_positions(elp_filepath, channel_names):
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
