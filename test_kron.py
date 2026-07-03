import numpy as np
import time
from scipy.sparse import csr_matrix, kron, csgraph

def build_graph_kron():
    N_ch = 60
    N_b = 5
    N_e = 40
    
    # Adjacency in epochs (tridiagonal)
    adj_e = csr_matrix(np.eye(N_e, k=-1) + np.eye(N_e, k=0) + np.eye(N_e, k=1), dtype=bool)
    # Adjacency in bands (tridiagonal)
    adj_b = csr_matrix(np.eye(N_b, k=-1) + np.eye(N_b, k=0) + np.eye(N_b, k=1), dtype=bool)
    
    # Adjacency in space (random sparse with self-loops)
    ch_adj_dense = np.random.rand(N_ch, N_ch) < 0.1
    np.fill_diagonal(ch_adj_dense, True)
    adj_ch = csr_matrix(ch_adj_dense)
    
    print("Construyendo matriz por Kron...")
    t0 = time.time()
    
    # (dest x src)
    adj_spatial = kron(adj_ch, adj_ch, format='csr')
    
    # (dest x src x band)
    adj_3d = kron(adj_spatial, adj_b, format='csr')
    
    # (dest x src x band x epoch)
    adj_4d = kron(adj_3d, adj_e, format='csr')
    
    print(f"Construida en {time.time()-t0:.4f} s. Dimensiones: {adj_4d.shape}, Non-zeros: {adj_4d.nnz}")
    
    # Simular clustering
    active = np.random.choice(adj_4d.shape[0], 10000, replace=False)
    active.sort()
    
    t0 = time.time()
    subgraph = adj_4d[active, :][:, active]
    n_comp, labels = csgraph.connected_components(subgraph, directed=False)
    print(f"Clustering en {time.time()-t0:.4f} s. Componentes: {n_comp}")
    
if __name__ == "__main__":
    build_graph_kron()
