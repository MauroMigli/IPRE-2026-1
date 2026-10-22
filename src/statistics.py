import numpy as np
from scipy.sparse import csr_matrix, kron, csgraph

def get_3d_positions(elp_filepath, channel_names):
    """Parses ELP file to get 3D coords for electrodes."""
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


def get_roi_3d_centroids(elp_filepath, rois_dict):
    """
    Calcula las posiciones 3D de los centroides de cada ROI/super-nodo
    promediando las coordenadas de sus electrodos constituyentes.
    
    Retorna:
    --------
    centroids_3d: np.ndarray de forma (n_rois, 3)
    roi_names: lista con los nombres de las ROIs en orden
    """
    roi_names = list(rois_dict.keys())
    all_channels = [ch for chs in rois_dict.values() for ch in chs]
    coords = get_3d_positions(elp_filepath, all_channels)
    ch_to_coord = {ch: coords[i] for i, ch in enumerate(all_channels)}
    
    centroids_3d = np.zeros((len(roi_names), 3))
    for r_idx, roi in enumerate(roi_names):
        member_coords = [ch_to_coord[ch] for ch in rois_dict[roi] if ch in ch_to_coord]
        if member_coords:
            centroids_3d[r_idx] = np.mean(member_coords, axis=0)
            
    return centroids_3d, roi_names


def get_spatial_adjacency_matrix(node_names, elp_file, R, rois_dict=None):
    """
    Construye la matriz de adyacencia espacial booleana NxN.
    Dos nodos están conectados si su distancia euclidiana es <= R.
    Soporta tanto canales individuales como ROIs/super-nodos.
    Si R es 0, la matriz será True solo en la diagonal (null).
    """
    node_names = list(node_names)
    if rois_dict is not None and all(name in rois_dict for name in node_names):
        centroids_3d, _ = get_roi_3d_centroids(elp_file, rois_dict)
        coords = centroids_3d
    else:
        coords = get_3d_positions(elp_file, node_names)
        
    n_nodes = len(node_names)
    adj = np.zeros((n_nodes, n_nodes), dtype=bool)
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= R:
                adj[i, j] = True
                    
    return adj


def build_edge_band_graph(ch_adj, n_bands):
    """
    Construye el grafo disperso de vecindad entre aristas dirigidas y bandas
    usando productos de Kronecker. Orden de las dimensiones (flatten order C):
    (dest, src, band). Dos aristas son vecinas si sus destinos y sus orígenes son
    espacialmente vecinos; dos bandas son vecinas si son contiguas.
    """
    adj_b = csr_matrix(np.eye(n_bands, k=-1) + np.eye(n_bands, k=0) + np.eye(n_bands, k=1), dtype=bool)
    adj_ch = csr_matrix(ch_adj, dtype=bool)
    
    adj_spatial = kron(adj_ch, adj_ch, format='csr')
    return kron(adj_spatial, adj_b, format='csr')


def compute_welch_t_map(D_A, D_B):
    """
    Calcula el estadístico T de Welch para dos muestras independientes con varianzas desiguales.
    D_A y D_B son arrays donde el eje 0 representa a los sujetos.
    Retorna el mapa T de la misma forma que D_A y D_B (sin el eje 0).
    """
    n1 = D_A.shape[0]
    n2 = D_B.shape[0]
    
    mean1 = np.mean(D_A, axis=0)
    mean2 = np.mean(D_B, axis=0)
    
    var1 = np.var(D_A, axis=0, ddof=1)
    var2 = np.var(D_B, axis=0, ddof=1)
    
    se_diff = np.sqrt(var1 / n1 + var2 / n2)
    
    t_stat = np.divide(mean1 - mean2, se_diff, out=np.zeros_like(mean1), where=se_diff!=0)
    t_stat = np.nan_to_num(t_stat)
    
    return np.abs(t_stat)  # Absoluto para 2 colas


def tfce_transform(T_map, adjacency, dh=0.1, E=0.5, H=2.0):
    """
    Aplica TFCE sobre un mapa T arbitrario cuya vecindad viene dada por un grafo
    disperso (ver build_edge_band_graph).
    """
    T_map = np.clip(T_map, 0, 20.0)
    
    tfce_map = np.zeros(T_map.size, dtype=float)
    max_t = np.max(T_map)
    if max_t <= 0:
        return tfce_map.reshape(T_map.shape)
        
    t_flat = T_map.ravel()
    for h in np.arange(dh, max_t + dh, dh):
        active_indices = np.nonzero(t_flat >= h)[0]
        if len(active_indices) == 0:
            continue
            
        subgraph = adjacency[active_indices, :][:, active_indices]
        _, labels = csgraph.connected_components(subgraph, directed=False)
        component_sizes = np.bincount(labels)
        tfce_map[active_indices] += (component_sizes[labels] ** E) * (h ** H) * dh
        
    return tfce_map.reshape(T_map.shape)


def fdrcorrect_bh(pvals):
    """
    Corrección de False Discovery Rate (FDR) usando el procedimiento de Benjamini-Hochberg.
    Devuelve los p-valores ajustados (q-valores).
    """
    pvals = np.asanyarray(pvals)
    n = len(pvals)
    if n == 0:
        return pvals
        
    sort_idx = np.argsort(pvals)
    pvals_sorted = pvals[sort_idx]
    
    adj_pvals = np.zeros(n)
    prev_q = 1.0
    for i in range(n - 1, -1, -1):
        q = pvals_sorted[i] * n / (i + 1)
        q = min(q, prev_q)
        adj_pvals[i] = q
        prev_q = q
        
    # Reordenar al orden original
    original_order_adj_pvals = np.zeros(n)
    original_order_adj_pvals[sort_idx] = adj_pvals
    
    return original_order_adj_pvals


def worker_permutation(seed, D_all, N_FT, adjacency, dh):
    """Worker para procesar una permutación Montecarlo en paralelo (estadístico máximo)."""
    rng = np.random.default_rng(seed)
    perm_indices = rng.permutation(len(D_all))
    D_FT_perm = D_all[perm_indices[:N_FT]]
    D_PT_perm = D_all[perm_indices[N_FT:]]
    
    t_map_perm = np.transpose(compute_welch_t_map(D_FT_perm, D_PT_perm), (1, 2, 0))  # (dest, src, band)
    return np.max(tfce_transform(t_map_perm, adjacency, dh=dh))
