import os
import sys
import unittest
import numpy as np

from reconstruction_engine.logger import PipelineLogger, PipelineTracker
from reconstruction_engine.step10_georeferencing import GeoreferencingEngine

class TestHelmertTransform(unittest.TestCase):
    def setUp(self):
        self.logger = PipelineLogger("Test-Helmert")
        self.tracker = PipelineTracker()
        self.engine = GeoreferencingEngine({}, self.logger, self.tracker)

    def test_exact_similarity_recovery(self):
        """Validates that 7-parameter Umeyama alignment recovers known scale, rotation, and translation."""
        # Generate random 3D source points
        rng = np.random.RandomState(42)
        X = rng.uniform(-100.0, 100.0, size=(20, 3))

        # Ground truth transformation
        true_scale = 2.45
        theta = np.radians(35.0)
        true_R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,             0,              1]
        ])
        true_t = np.array([1250.5, 3420.2, 85.0])

        # Apply transformation to get target points Y
        Y = (true_scale * (true_R @ X.T) + true_t[:, None]).T

        weights = np.ones(len(X))
        T_mat, rec_s, rec_R, rec_t = self.engine._umeyama_alignment(X, Y, weights)

        # Assert scale recovered within 1e-4
        self.assertAlmostEqual(rec_s, true_scale, places=4, msg="Recovered scale deviates from ground truth!")

        # Assert rotation recovered within 1e-4
        np.testing.assert_allclose(rec_R, true_R, atol=1e-4, err_msg="Recovered rotation deviates from ground truth!")

        # Assert translation recovered within 1e-3
        np.testing.assert_allclose(rec_t, true_t, atol=1e-3, err_msg="Recovered translation deviates from ground truth!")

        # Assert zero residual
        transformed = (rec_s * (rec_R @ X.T) + rec_t[:, None]).T
        residual = np.linalg.norm(transformed - Y, axis=1)
        self.assertLess(float(np.mean(residual)), 1e-4, "Residual error is not near zero on noiseless data!")

if __name__ == "__main__":
    unittest.main()
