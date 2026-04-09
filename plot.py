import plotly.graph_objects as go
import plotly.offline as pyo
import numpy as np

def export_interactive_3d_network(coords_3d, p_values, channel_names, filename="red_conectividad_3d.html"):
    print(f"\n--- Generando visualización interactiva en {filename} ---")
    
    p_threshold = 0.05 /2070
    highly_sig_threshold = 0.01 / 2070
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
                # El origen es j, el destino es i (basado en ij dDTF_dest_src)
                x_src, y_src, z_src = coords_3d[j]
                x_dest, y_dest, z_dest = coords_3d[i]
                
                # Asignación de escala de color según el p-valor (rojo para mayor significancia)
                if p_values[i, j] < highly_sig_threshold:
                    color = 'red'
                    width = 4
                    color_val = 1 # Para los conos
                else:
                    color = 'orange'
                    width = 2
                    color_val = 0
                
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
                
                # Posicionar la punta de la flecha al 75% del trayecto para no tapar el nodo destino
                cone_x.append(x_src + u * 0.75)
                cone_y.append(y_src + v * 0.75)
                cone_z.append(z_src + w * 0.75)
                cone_u.append(u)
                cone_v.append(v)
                cone_w.append(w)
                cone_colors.append(color_val)

    # Trazado de las puntas de flecha
    if cone_x:
        arrows_trace = go.Cone(
            x=cone_x, y=cone_y, z=cone_z,
            u=cone_u, v=cone_v, w=cone_w,
            sizemode="absolute",
            sizeref=0.5,
            anchor="tip",
            colorscale=[[0, 'orange'], [1, 'red']],
            cmin=0, cmax=1,
            showscale=False,
            hoverinfo='none'
        )
        edges_traces.append(arrows_trace)

    layout = go.Layout(
        title=f"Red Direccional Significativa (p < {p_threshold})<br>Rojo: p < {highly_sig_threshold}",
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
