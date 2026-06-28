import numpy as np
from scipy import stats
from scipy.ndimage import label, generate_binary_structure

def compute_t_map(D_array):
    """
    Calcula el estadístico T para 1 muestra a lo largo del eje 0 (muestras/instancias).
    D_array: array de shape (N_samples, n_epoch, n_freq, n_dest, n_src) o cualquier shape,
             donde el eje 0 representa las distintas muestras (instancias del diseño 2x2).
    Retorna:
    T_map: array con la misma shape que D_array pero sin el eje 0.
    """
    # T(i, j, f, t) = \bar D(i, j, f, t) / SE[D(i, j, f, t)]
    mean_D = np.mean(D_array, axis=0)
    # ddof=1 para desviación estándar muestral
    std_D = np.std(D_array, axis=0, ddof=1) 
    
    n_samples = D_array.shape[0]
    se_D = std_D / np.sqrt(n_samples)
    
    # Evitar división por cero
    t_stat = np.divide(mean_D, se_D, out=np.zeros_like(mean_D), where=se_D!=0)
    
    # Rellenar posibles NaNs
    t_stat = np.nan_to_num(t_stat)
    return t_stat

def get_structuring_element(spatial_adjacency='null'):
    """
    Devuelve el elemento estructurante para ndimage.label.
    Para el espacio 4D (dest, src, freq, epoch).
    """
    # Empezamos con conectividad en freq y epoch
    # 3x3 para 8-connectivity en 2D (freq, epoch)
    struct_2d = np.ones((3, 3), dtype=bool)
    
    if spatial_adjacency == 'null':
        # Conectividad espacial nula: un nodo (i, j) solo se conecta consigo mismo (i, j).
        # El elemento estructurante 4D tiene tamaño 3 en todas las dimensiones.
        # Solo en el centro espacial (1, 1) copiamos la conectividad 2D.
        struct_4d = np.zeros((3, 3, 3, 3), dtype=bool)
        struct_4d[1, 1, :, :] = struct_2d
    elif spatial_adjacency == 'total':
        # Conectividad espacial total: un nodo se conecta a cualquier otro (i', j').
        # Como ndimage.label busca vecinos locales en la cuadrícula, para 'total' 
        # es mejor resolverlo analíticamente proyectando a 2D, ya que ndimage.label 
        # no soporta 'todos con todos' sin construir una matriz densa gigante.
        # Por tanto, para 'total' no usaremos un elemento estructurante 4D estándar,
        # sino que lo manejaremos en la lógica principal.
        struct_4d = None
    else:
        raise ValueError("spatial_adjacency debe ser 'total' o 'null'")
        
    return struct_4d

def tfce_transform(T_map, spatial_adjacency='null', dh=0.1, E=0.5, H=2.0):
    """
    Aplica la transformación TFCE a un mapa de estadísticos 4D.
    
    T_map: array 4D (dest, src, freq, epoch)
    spatial_adjacency: 'total' o 'null'
    dh: paso de discretización de la integral
    """
    tfce_map = np.zeros_like(T_map, dtype=float)
    
    # Solo procesamos valores positivos para la cola derecha (se puede adaptar para dos colas)
    max_t = np.max(T_map)
    if max_t <= 0:
        return tfce_map
        
    # Rango de integración (Suma de Riemann)
    hs = np.arange(dh, max_t + dh, dh)
    
    if spatial_adjacency == 'null':
        struct = get_structuring_element('null')
        
    for h in hs:
        # V_h es la máscara suprathreshold
        V_h = T_map >= h
        
        # Array para almacenar la extensión e(h, p) para cada punto p
        e_h = np.zeros_like(T_map, dtype=float)
        
        if spatial_adjacency == 'null':
            # Ejecutamos label en 4D con conectividad aislada espacialmente
            labeled_array, num_features = label(V_h, structure=struct)
            
            if num_features > 0:
                # Contar los tamaños de cada componente
                # bincount es muy rápido para esto
                component_sizes = np.bincount(labeled_array.ravel())
                
                # Asignar a cada punto el tamaño de su componente
                # labeled_array tiene valores de 0 (background) a num_features
                e_h = component_sizes[labeled_array]
                # El fondo no pertenece a ningún clúster, su tamaño debe ser 0
                e_h[labeled_array == 0] = 0
                
        elif spatial_adjacency == 'total':
            # Proyección 2D (freq, epoch)
            # Un punto 2D está activo si al menos una arista lo está
            V_h_2d = np.any(V_h, axis=(0, 1))
            
            # Elemento estructurante 2D (8-connectivity)
            struct_2d = np.ones((3, 3), dtype=bool)
            labeled_2d, num_features = label(V_h_2d, structure=struct_2d)
            
            if num_features > 0:
                # El tamaño del componente 4D es la suma total de elementos activos en V_h 
                # que caen dentro del componente 2D.
                
                # Expandimos labeled_2d a 4D para alinear con V_h
                # shape actual: (n_freq, n_epoch). Nueva shape: (1, 1, n_freq, n_epoch)
                labeled_4d = labeled_2d[np.newaxis, np.newaxis, :, :]
                
                # Solo los puntos activos en V_h forman parte del clúster
                # Etiquetamos cada punto en V_h con la etiqueta de su proyección 2D
                active_labels = labeled_4d * V_h
                
                # Contamos cuántos elementos activos hay por cada etiqueta
                component_sizes = np.bincount(active_labels.ravel())
                
                # Asignamos a e_h el tamaño correspondiente
                e_h = component_sizes[active_labels]
                e_h[active_labels == 0] = 0

        # Acumular la suma de Riemann: e(h, p)^E * h^H * dh
        tfce_map += (e_h ** E) * (h ** H) * dh
        
    return tfce_map

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
    
    return np.abs(t_stat)  # Usamos el valor absoluto para contrastes bidireccionales en TFCE
