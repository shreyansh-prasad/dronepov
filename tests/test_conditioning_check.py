import os
import sys
import unittest
import numpy as np

from reconstruction_engine.logger import PipelineLogger, PipelineTracker
from reconstruction_engine.step02_sparse_sfm import SparseSfMEngine

class TestConditioningCheck(unittest.TestCase):
    def setUp(self):
        self.logger = PipelineLogger("Test-Conditioning")
        self.tracker = PipelineTracker()
        self.config = {
            "step02_sparse_sfm": {
                "conditioning_threshold": 0.05
            }
        }
        self.engine = SparseSfMEngine(self.config, self.logger, self.tracker, {})

    def test_linear_flight_trajectory_detected(self):
        """Validates that a near-linear single-pass flight path is correctly flagged as poorly conditioned."""
        # 10 cameras flying in a straight line along the X axis
        linear_poses = {}
        for i in range(10):
            linear_poses[f"frame_{i:04d}.jpg"] = {
                "center": np.array([float(i * 10.0), 0.0, 50.0]) # Straight line with slight noise
            }

        ratio, status, warning = self.engine._check_trajectory_conditioning(linear_poses)
        self.assertEqual(status, "POORLY_CONDITIONED_NEAR_LINEAR")
        self.assertIsNotNone(warning)
        self.assertLess(ratio, 0.05)

    def test_well_conditioned_flight_trajectory(self):
        """Validates that a trajectory with 2D/3D distribution (e.g. cross-grid or looped) is marked WELL_CONDITIONED."""
        # Cameras distributed in a loop or grid
        grid_poses = {}
        idx = 0
        for x in [0.0, 20.0, 40.0]:
            for y in [0.0, 20.0, 40.0]:
                grid_poses[f"frame_{idx:04d}.jpg"] = {
                    "center": np.array([x, y, 50.0 + np.sin(x)])
                }
                idx += 1

        ratio, status, warning = self.engine._check_trajectory_conditioning(grid_poses)
        self.assertEqual(status, "WELL_CONDITIONED")
        self.assertIsNone(warning)
        self.assertGreater(ratio, 0.05)

if __name__ == "__main__":
    unittest.main()
