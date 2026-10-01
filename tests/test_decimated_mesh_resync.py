# test_decimated_mesh_resync.py
"""Unit test for ExporterReporterEngine._export_decimated_mesh resynchronization.
This version uses a minimal Open3D mock that implements a real brute‑force
KD‑Tree nearest‑neighbor search, ensuring the attribute‑resync logic is
exercised correctly.
"""
import os
import sys
import tempfile
import shutil
import numpy as np
import importlib.util

# Load the Open3D mock from the Antigravity artifact directory and register it as "open3d"
import pathlib
# Locate the Open3D mock in the Antigravity artifact directory
artifact_dir = pathlib.Path.home() / ".gemini" / "antigravity-ide" / "brain" / "9b391322-4159-46d4-ab68-b9d80aba14f7"
mock_path = artifact_dir / "open3d_mock.py"
spec = importlib.util.spec_from_file_location("open3d", str(mock_path))
open3d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(open3d)
sys.modules["open3d"] = open3d

# Minimal stub logger
class StubLogger:
    def __init__(self):
        self.messages = []
    def info(self, msg):
        self.messages.append(("info", msg))
    def warning(self, msg):
        self.messages.append(("warning", msg))
    def success(self, msg):
        self.messages.append(("success", msg))
    def stage_header(self, stage, title):
        self.messages.append(("stage", stage, title))

# Minimal stub tracker
class StubTracker:
    def __init__(self):
        self.metrics = {}
        self.stage_timings_sec = {}
        self.stage_status = {}
        self.degraded_reasons = {}
    def to_report_dict(self):
        return {
            "metrics": self.metrics,
            "stage_timings_sec": self.stage_timings_sec,
            "stage_status": self.stage_status,
            "degraded_reasons": self.degraded_reasons,
        }

def test_export_decimated_mesh_resync():
    # Create a simple square mesh (2 triangles)
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(vertices)
    mesh.triangles = open3d.utility.Vector3iVector(triangles)
    # Assign vertex colors
    colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    mesh.vertex_colors = open3d.utility.Vector3dVector(colors)
    # Dummy densities and provenance
    densities = np.array([0.9, 0.8, 0.85, 0.95], dtype=np.float32)
    provenance = np.array(["mvs", "ai", "mvs", "ai"], dtype=object)

    # Write OBJ with a duplicated vertex to simulate mismatch
    tmp_dir = tempfile.mkdtemp()
    obj_path = os.path.join(tmp_dir, "mesh_mismatch.obj")
    open3d.io.write_triangle_mesh(obj_path, mesh)
    # Duplicate the first vertex line in the OBJ file
    with open(obj_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    v_lines = [i for i, l in enumerate(lines) if l.startswith('v ')]
    if v_lines:
        first_v = lines[v_lines[0]]
        lines.insert(v_lines[0] + 1, first_v)
    with open(obj_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # Prepare ExporterReporterEngine
    from reconstruction_engine.step11_exporter_reporter import ExporterReporterEngine
    config = {"step11_export": {"decimation_factor": 0.5}}
    logger = StubLogger()
    tracker = StubTracker()
    engine = ExporterReporterEngine(config, logger, tracker)

    out_glb = os.path.join(tmp_dir, "decimated.glb")
    # Run export decimation with OBJ path for resync
    engine._export_decimated_mesh(mesh, out_glb, densities, provenance, obj_path=obj_path)

    # Retrieve the decimated mesh directly from the mock's store (no actual file I/O)
    decimated = open3d.io.read_triangle_mesh(out_glb)
    assert hasattr(decimated, "vertex_attributes"), "Vertex attributes missing"
    assert "densities" in decimated.vertex_attributes, "Densities not baked"
    assert "provenance" in decimated.vertex_attributes, "Provenance not baked"
    # Ensure UVs are not present
    assert not hasattr(decimated, "triangle_uvs"), "UVs should have been removed"
    # Densities array length should match decimated vertex count
    assert len(decimated.vertex_attributes["densities"]) == len(decimated.vertices), "Density length mismatch"
    # Provenance array length should match decimated vertex count
    assert len(decimated.vertex_attributes["provenance"]) == len(decimated.vertices), "Provenance length mismatch"

    # Clean up temporary directory
    shutil.rmtree(tmp_dir)
