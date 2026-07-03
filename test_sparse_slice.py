import numpy as np
import time
from scipy.sparse import csr_matrix, csgraph

# Simular 3600 edges, 5 bands, 40 epochs = 720,000 nodos
N = 720000
# Asumimos 50 vecinos por nodo
rows = np.random.randint(0, N, N * 50)
cols = np.random.randint(0, N, N * 50)
data = np.ones(N * 50, dtype=bool)

print("Construyendo matriz dispersa...")
t0 = time.time()
full_graph = csr_matrix((data, (rows, cols)), shape=(N, N))
print(f"Matriz construida en {time.time()-t0:.2f} s")

# Simular voxels activos (ej. 10,000 activos)
active_indices = np.random.choice(N, 10000, replace=False)
active_indices.sort()

print("Filtrando subgrafo y buscando componentes conexas...")
t0 = time.time()
# Slicing the graph
subgraph = full_graph[active_indices, :][:, active_indices]
n_comp, labels = csgraph.connected_components(subgraph, directed=False)
print(f"Completado en {time.time()-t0:.4f} s. Componentes: {n_comp}")
