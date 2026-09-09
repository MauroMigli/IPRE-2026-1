import unittest
import numpy as np
import os
import sys

import parameters
from src.statistics import get_roi_3d_centroids, get_spatial_adjacency_matrix, compute_welch_t_map, fdrcorrect_bh, tfce_transform, build_4d_graph
from src.connectivity import aggregate_channels_to_rois

class TestROIsAndMultivariate(unittest.TestCase):
    def setUp(self):
        # 46 canales supervivientes
        self.ch_names = [
            'E2', 'E3', 'E4', 'E6', 'E7', 'E9', 'E11', 'E12', 'E13', 'E14',
            'E15', 'E16', 'E18', 'E19', 'E20', 'E21', 'E22', 'E24', 'E25', 'E26',
            'E27', 'E28', 'E30', 'E31', 'E33', 'E34', 'E36', 'E38', 'E40', 'E41',
            'E42', 'E44', 'E45', 'E46', 'E48', 'E49', 'E50', 'E51', 'E52', 'E53',
            'E54', 'E56', 'E57', 'E58', 'E59', 'E60'
        ]
        self.n_channels = len(self.ch_names)
        self.n_epochs = 5
        self.n_times = 250
        np.random.seed(42)
        self.mock_data = np.random.randn(self.n_epochs, self.n_channels, self.n_times)

    def test_roi_centroids_geometry(self):
        centroids, names = get_roi_3d_centroids(parameters.ELP_FILE, parameters.ROIS)
        self.assertEqual(len(names), 8)
        self.assertEqual(centroids.shape, (8, 3))
        
        # Verificar distancia inter-centroides mínima (> 6.0 cm)
        min_dist = float('inf')
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = np.linalg.norm(centroids[i] - centroids[j])
                if d < min_dist:
                    min_dist = d
        self.assertGreater(min_dist, 6.0, "La distancia mínima inter-centroide debe ser mayor a 6 cm")

    def test_aggregate_channels_mean(self):
        roi_data = aggregate_channels_to_rois(self.mock_data, self.ch_names, parameters.ROIS, method='mean')
        self.assertEqual(roi_data.shape, (self.n_epochs, 8, self.n_times))
        self.assertFalse(np.isnan(roi_data).any())
        self.assertFalse(np.isinf(roi_data).any())

    def test_aggregate_channels_pca(self):
        roi_data = aggregate_channels_to_rois(self.mock_data, self.ch_names, parameters.ROIS, method='pca')
        self.assertEqual(roi_data.shape, (self.n_epochs, 8, self.n_times))
        self.assertFalse(np.isnan(roi_data).any())
        self.assertFalse(np.isinf(roi_data).any())

    def test_spatial_adjacency_rois(self):
        adj = get_spatial_adjacency_matrix(parameters.ROI_NAMES, parameters.ELP_FILE, R=9.5, rois_dict=parameters.ROIS)
        self.assertEqual(adj.shape, (8, 8))
        self.assertTrue(np.all(np.diag(adj)))  # Diagonal debe ser True
        # Debe haber conexiones con vecinos cercanos a R=9.5 cm
        self.assertGreater(np.sum(adj), 8)

    def test_statistics_roi_tensor(self):
        # Probar t-test, FDR y TFCE sobre tensores con dimensiones de ROIs
        n_rois = 8
        n_bands = 5
        n_ep = 4
        
        # Simular dos grupos FT y PT con diferencia inducida en un enlace específico
        D_FT = np.random.randn(10, n_ep, n_bands, n_rois, n_rois) * 0.1
        D_PT = np.random.randn(10, n_ep, n_bands, n_rois, n_rois) * 0.1
        # Inyectar diferencia fuerte en conexión ROI 0 -> ROI 1 en banda 0
        D_FT[:, :, 0, 0, 1] += 1.5

        t_map = compute_welch_t_map(D_FT, D_PT)
        self.assertEqual(t_map.shape, (n_ep, n_bands, n_rois, n_rois))
        
        # Mapa 4D para TFCE: (dest, src, band, epoch)
        T_4d = np.transpose(t_map, (2, 3, 1, 0))
        adj_spatial = get_spatial_adjacency_matrix(parameters.ROI_NAMES, parameters.ELP_FILE, R=9.5, rois_dict=parameters.ROIS)
        adj_4d = build_4d_graph(adj_spatial, n_bands=n_bands, n_epochs=n_ep)
        
        tfce = tfce_transform(T_4d, spatial_adjacency=adj_4d, dh=0.5)
        self.assertEqual(tfce.shape, T_4d.shape)
        self.assertGreater(tfce[0, 1, 0, 0], 0.0)

if __name__ == '__main__':
    unittest.main()
