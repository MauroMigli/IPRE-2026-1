import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import parameters
from src.preprocessing import downsample_epochs
from src.connectivity import build_lagged_design, fit_mvar_multitrial, ddtf_from_mvar
from src.model_order import evaluate_orders
from src.statistics import get_spatial_adjacency_matrix, build_edge_band_graph, tfce_transform, fdrcorrect_bh


def simulate_chain(n_epochs=40, n_times=150, seed=0):
    """Cadena 0 -> 1 -> 2 (rezagos 1 y 2): no existe un vínculo directo 0 -> 2."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_epochs, 3, n_times))
    for ep in range(n_epochs):
        e = rng.standard_normal((3, n_times))
        for t in range(2, n_times):
            x[ep, 0, t] = 0.5 * x[ep, 0, t-1] + e[0, t]
            x[ep, 1, t] = 0.4 * x[ep, 1, t-1] + 0.6 * x[ep, 0, t-1] + e[1, t]
            x[ep, 2, t] = 0.4 * x[ep, 2, t-1] + 0.6 * x[ep, 1, t-2] + e[2, t]
    return x


class TestConnectivity(unittest.TestCase):
    def test_lagged_design_does_not_cross_epochs(self):
        x = np.arange(2 * 1 * 6, dtype=float).reshape(2, 1, 6)  # época 0: 0..5, época 1: 6..11
        Y, X = build_lagged_design(x, p=2)
        self.assertEqual(Y.shape, (8, 1))
        # Primera respuesta de la época 1 (valor 8) debe regresarse sobre 7 y 6, no sobre 5
        self.assertTrue(np.allclose(X[4], [1, 7, 6]))

    def test_mvar_recovers_coefficients(self):
        fit = fit_mvar_multitrial(simulate_chain(), p=2)
        self.assertAlmostEqual(fit['A'][0][1, 0], 0.6, delta=0.05)
        self.assertAlmostEqual(fit['A'][1][2, 1], 0.6, delta=0.05)
        self.assertLess(fit['max_root'], 1.0)

    def test_ddtf_suppresses_indirect_link(self):
        fit = fit_mvar_multitrial(simulate_chain(), p=3)
        d = ddtf_from_mvar(fit['A'], fit['V'], 20.0, freqs=np.arange(0.5, 10.0, 0.5)).mean(axis=0)
        self.assertGreater(d[1, 0], 0.2)   # directo 0 -> 1
        self.assertGreater(d[2, 1], 0.2)   # directo 1 -> 2
        self.assertLess(d[2, 0], 0.05)     # indirecto 0 -> 2 suprimido por la coherencia parcial
        self.assertTrue(np.allclose(np.diagonal(d), 0))

    def test_bic_recovers_order(self):
        res = evaluate_orders(simulate_chain(), max_p=8)
        self.assertEqual(res['lags'][np.argmin(res['bic'])], 2)

    def test_downsample_length(self):
        x = np.random.default_rng(1).standard_normal((3, 8, 1500))
        self.assertEqual(downsample_epochs(x, 500.0, parameters.FS_TARGET).shape[-1], int(1500 * parameters.FS_TARGET / 500))


class TestStatistics(unittest.TestCase):
    def test_fdr_matches_reference(self):
        from statsmodels.stats.multitest import fdrcorrection
        p = np.random.default_rng(2).uniform(size=112) ** 2
        self.assertTrue(np.allclose(fdrcorrect_bh(p), fdrcorrection(p)[1]))

    def test_tfce_shape_and_monotonic(self):
        adj = get_spatial_adjacency_matrix(parameters.ROI_NAMES, parameters.ELP_FILE, parameters.R_ROI_DEFAULT, rois_dict=parameters.ROIS)
        graph = build_edge_band_graph(adj, n_bands=2)
        t = np.zeros((8, 8, 2))
        t[1, 0, 0] = 3.0
        t[2, 0, 0] = 1.0
        tfce = tfce_transform(t, graph)
        self.assertEqual(tfce.shape, t.shape)
        self.assertGreater(tfce[1, 0, 0], tfce[2, 0, 0])
        self.assertEqual(tfce[5, 5, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
