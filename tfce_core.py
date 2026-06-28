import numpy as np
from scipy.ndimage import label
from scipy.sparse import csr_matrix, kron, csgraph
from html_plotter import get_3d_positions

def get_spatial_adjacency_matrix(ch_names, elp_file, R):
    """
    Construye la matriz de adyacencia espacial booleana NxN.
    Dos canales están conectados si su distancia euclidiana es <= R.
    Si R es 0, la matriz será True solo en la diagonal (null).
    """
    coords = get_3d_positions(elp_file, ch_names)
    n_ch = len(ch_names)
    adj = np.zeros((n_ch, n_ch), dtype=bool)
    
    for i in range(n_ch):
        for j in range(n_ch):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= R:
                adj[i, j] = True
                    
    return adj

def build_4d_graph(ch_adj, n_bands, n_epochs):
    """
    Construye el grafo disperso completo de 4D usando productos de Kronecker.
    Las dimensiones ordenadas (flatten order C) son:
    (dest, src, band, epoch)
    """
    adj_e = csr_matrix(np.eye(n_epochs, k=-1) + np.eye(n_epochs, k=0) + np.eye(n_epochs, k=1), dtype=bool)
    adj_b = csr_matrix(np.eye(n_bands, k=-1) + np.eye(n_bands, k=0) + np.eye(n_bands, k=1), dtype=bool)
    adj_ch = csr_matrix(ch_adj, dtype=bool)
    
    # Kronecker order: dest -> src -> band -> epoch
    adj_spatial = kron(adj_ch, adj_ch, format='csr')
    adj_3d = kron(adj_spatial, adj_b, format='csr')
    adj_4d = kron(adj_3d, adj_e, format='csr')
    
    return adj_4d

def compute_t_map(D_array):
    """(Código de prueba T de una muestra, obsoleto pero mantenido por compatibilidad)"""
    mean_D = np.mean(D_array, axis=0)
    std_D = np.std(D_array, axis=0, ddof=1) 
    n_samples = D_array.shape[0]
    se_D = std_D / np.sqrt(n_samples)
    t_stat = np.divide(mean_D, se_D, out=np.zeros_like(mean_D), where=se_D!=0)
    t_stat = np.nan_to_num(t_stat)
    return t_stat

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

def get_structuring_element(spatial_adjacency='null'):
    struct_2d = np.ones((3, 3), dtype=bool)
    if spatial_adjacency == 'null':
        struct_4d = np.zeros((3, 3, 3, 3), dtype=bool)
        struct_4d[1, 1, :, :] = struct_2d
    elif spatial_adjacency == 'total':
        struct_4d = None
    else:
        raise ValueError("spatial_adjacency debe ser 'total' o 'null'")
    return struct_4d

def tfce_transform(T_map, spatial_adjacency='null', dh=0.1, E=0.5, H=2.0):
    """
    Aplica TFCE.
    Si spatial_adjacency es un string ('total', 'null'), usa ndimage.label (método legado 2D).
    Si spatial_adjacency es un csr_matrix (Grafo de Kronecker), usa csgraph.connected_components (Nuevo).
    """
    # Protegemos contra outliers estadísticos (ej. canales con varianza casi cero en permutaciones)
    # Un valor T > 20.0 es astronómicamente significativo (p < 1e-15).
    # Limitarlo a 20 evita que el bucle de integrales 'hs' intente dar miles de pasos y congele el servidor.
    T_map = np.clip(T_map, 0, 20.0)
    
    tfce_map = np.zeros_like(T_map, dtype=float)
    max_t = np.max(T_map)
    if max_t <= 0:
        return tfce_map
        
    hs = np.arange(dh, max_t + dh, dh)
    
    use_sparse_graph = isinstance(spatial_adjacency, csr_matrix)
    
    if not use_sparse_graph and spatial_adjacency == 'null':
        struct = get_structuring_element('null')
        
    for h in hs:
        V_h = T_map >= h
        e_h = np.zeros_like(T_map, dtype=float)
        
        if use_sparse_graph:
            # 1. Encontrar los índices (1D flatten) de los nodos activos
            active_mask = V_h.ravel()
            active_indices = np.nonzero(active_mask)[0]
            
            if len(active_indices) > 0:
                # 2. Filtrar subgrafo (Ojo: esto es rápido gracias a CSR slicing)
                subgraph = spatial_adjacency[active_indices, :][:, active_indices]
                
                # 3. Detectar clústeres conexos
                num_features, labels = csgraph.connected_components(subgraph, directed=False)
                
                # 4. Asignar tamaños de clúster
                if num_features > 0:
                    component_sizes = np.bincount(labels)
                    e_h_flat = np.zeros_like(active_mask, dtype=float)
                    e_h_flat[active_indices] = component_sizes[labels]
                    e_h = e_h_flat.reshape(T_map.shape)
                    
        else:
            # Legacy code para 'null' y 'total' puros
            if spatial_adjacency == 'null':
                labeled_array, num_features = label(V_h, structure=struct)
                if num_features > 0:
                    component_sizes = np.bincount(labeled_array.ravel())
                    e_h = component_sizes[labeled_array]
                    e_h[labeled_array == 0] = 0
            elif spatial_adjacency == 'total':
                V_h_2d = np.any(V_h, axis=(0, 1))
                struct_2d = np.ones((3, 3), dtype=bool)
                labeled_2d, num_features = label(V_h_2d, structure=struct_2d)
                if num_features > 0:
                    labeled_4d = labeled_2d[np.newaxis, np.newaxis, :, :]
                    active_labels = labeled_4d * V_h
                    component_sizes = np.bincount(active_labels.ravel())
                    e_h = component_sizes[active_labels]
                    e_h[active_labels == 0] = 0

        # Acumular la suma de Riemann
        tfce_map += (e_h ** E) * (h ** H) * dh
        
    return tfce_map
