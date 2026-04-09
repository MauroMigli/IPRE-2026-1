import plotly.graph_objects as go
import plotly.offline as pyo
import numpy as np

def export_interactive_3d_network(coords_3d, p_values, channel_names, filename="red_conectividad_3d.html"):
    print(f"\n--- Generando visualización interactiva en {filename} ---")
    
    p_threshold = 0.05
    highly_sig_threshold = 0.01
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
    
    for i in range(n_ch):
        for j in range(n_ch):
            if i != j and p_values[i, j] < p_threshold:
                x_src, y_src, z_src = coords_3d[j]
                x_dest, y_dest, z_dest = coords_3d[i]
                
                if p_values[i, j] < highly_sig_threshold:
                    color = 'red'
                    width = 4
                else:
                    color = 'orange'
                    width = 2
                
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

    layout = go.Layout(
        title=f"Red de Conectividad Direccional Significativa (ANOVA p < {p_threshold})",
        scene=dict(
            xaxis=dict(title='X (cm)', showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title='Y (cm)', showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(title='Z (cm)', showgrid=False, zeroline=False, showticklabels=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
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
