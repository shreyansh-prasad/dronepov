import os
import sys
import unittest
import numpy as np
import open3d as o3d

from reconstruction_engine.logger import PipelineLogger, PipelineTracker
from reconstruction_engine.step09_confidence_tagging import ConfidenceTaggingEngine

class TestConfidenceWeighting(unittest.TestCase):
    def setUp(self):
        self.logger = PipelineLogger("Test-Confidence")
        self.tracker = PipelineTracker()
        self.config = {
            "mvs_base_confidence": 0.90,
            "ai_depth_max_ceiling": 0.45,
            "triangulation_angle_weight": 0.35,
            "density_weight": 0.30
        }
        self.engine = ConfidenceTaggingEngine(self.config, self.logger, self.tracker)

    def test_provenance_hard_ceiling_rule(self):
        """Validates that points/vertices derived from AI depth strictly obey the confidence ceiling."""
        # Create small synthetic mesh (box)
        mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        num_verts = len(mesh.vertices)
        
        # Half vertices are MVS, half are AI-depth
        provenance = np.array(["mvs"] * (num_verts // 2) + ["ai_depth"] * (num_verts - num_verts // 2), dtype=object)
        densities = np.ones(num_verts, dtype=np.float32)

        # Mock camera poses
        camera_poses = {
            "cam1": {"center": np.array([0.0, 0.0, 5.0])},
            "cam2": {"center": np.array([2.0, 0.0, 5.0])}
        }

        os.makedirs("tests/scratch", exist_ok=True)
        res = self.engine.compute_and_export_confidence(
            mesh=mesh,
            densities=densities,
            provenance=provenance,
            camera_poses=camera_poses,
            workspace_dir="tests/scratch",
            output_prefix="test_conf"
        )

        conf_arr = res["confidence_array"]
        ai_indices = np.where(provenance == "ai_depth")[0]
        mvs_indices = np.where(provenance == "mvs")[0]

        # AI-depth vertices must have significantly lower confidence than MVS vertices
        self.assertLessEqual(
            float(np.mean(conf_arr[ai_indices])),
            0.60,
            "AI-depth confidence ceiling violated!"
        )
        self.assertGreater(
            float(np.mean(conf_arr[mvs_indices])),
            float(np.mean(conf_arr[ai_indices])),
            "MVS points must have higher confidence than AI-depth fallback points!"
        )

if __name__ == "__main__":
    unittest.main()
