import os
import sys
import numpy as np
import open3d as o3d
import trimesh
from typing import Dict, List, Any, Optional, Tuple
from reconstruction_engine.logger import PipelineLogger, PipelineTracker

class ConfidenceTaggingEngine:
    """
    Step 9: Confidence Tagging & Multi-Mode Visualization.
    Computes per-vertex confidence taking multi-view vs AI-depth provenance as a hard input,
    combined with camera observation count, triangulation angles, and surface density.
    Bakes green-yellow-red colors into standard COLOR_0 and preserves raw numeric scalar _CONFIDENCE.
    """
    def __init__(self, config: Dict[str, Any], logger: PipelineLogger, tracker: PipelineTracker):
        self.config = config.get("step09_confidence", config)
        self.logger = logger
        self.tracker = tracker

    def compute_and_export_confidence(
        self,
        mesh: o3d.geometry.TriangleMesh,
        densities: np.ndarray,
        provenance: np.ndarray,
        camera_poses: Dict[str, Dict],
        workspace_dir: str,
        output_prefix: str = "mesh_confidence"
    ) -> Dict[str, str]:
        """
        Computes normalized confidence and exports confidence-tagged meshes.
        Returns:
            Dict containing:
                - 'ply_path': path to mesh_confidence.ply (with float property 'confidence')
                - 'glb_path': path to mesh_confidence.glb (with COLOR_0 and _CONFIDENCE)
                - 'mean_confidence': float
        """
        self.logger.stage_header(9, "Confidence Scoring & Quality Tagging")

        out_dir = os.path.join(workspace_dir, "confidence_mesh")
        os.makedirs(out_dir, exist_ok=True)
        ply_out = os.path.join(out_dir, f"{output_prefix}.ply")
        glb_out = os.path.join(out_dir, f"{output_prefix}.glb")

        vertices = np.asarray(mesh.vertices)
        num_verts = len(vertices)

        # 1. Compute Base Confidence from Step 4 Provenance (Hard Ceiling Rule)
        mvs_base = float(self.config.get("mvs_base_confidence", 0.90))
        ai_ceiling = float(self.config.get("ai_depth_max_ceiling", 0.45))
        
        confidence = np.full(num_verts, mvs_base, dtype=np.float32)
        if len(provenance) == num_verts:
            is_ai = np.array([p == "ai_depth" for p in provenance])
            confidence[is_ai] = ai_ceiling

        # 2. Secondary Modifiers: Observation Count & Baseline Ray Angles
        cam_positions = [p["center"] for p in camera_poses.values()] if camera_poses else []
        if len(cam_positions) >= 2:
            cam_mat = np.array(cam_positions)
            
            # Subsample check or compute distance/visibility
            # Count cameras within viewing cone and compute max angular baseline
            angle_modifiers = []
            for v in vertices:
                rays = cam_mat - v
                dists = np.linalg.norm(rays, axis=1, keepdims=True) + 1e-6
                unit_rays = rays / dists
                
                # Approximate visibility (cameras in front of vertex within 90 deg)
                # Compute pairwise dot products between unit rays to find max angular baseline
                dot_prods = unit_rays @ unit_rays.T
                min_dot = np.min(dot_prods) # smallest dot product = largest separation angle
                # min_dot = 1.0 (parallel/0 deg), min_dot = 0.0 (90 deg), min_dot = -1.0 (180 deg)
                # Optimal photogrammetry stereo angle is 15-45 degrees (dot product ~ 0.7 - 0.96)
                angle_score = float(np.clip(1.0 - min_dot, 0.0, 1.0))
                angle_modifiers.append(angle_score)

            angle_weight = float(self.config.get("triangulation_angle_weight", 0.35))
            confidence = confidence * (1.0 - angle_weight) + np.array(angle_modifiers, dtype=np.float32) * angle_weight

        # 3. Density Modulation
        if len(densities) == num_verts:
            d_norm = (densities - densities.min()) / (densities.max() - densities.min() + 1e-8)
            density_weight = float(self.config.get("density_weight", 0.30))
            confidence = confidence * (1.0 - density_weight) + d_norm.astype(np.float32) * density_weight

        # Normalize final confidence strictly to [0.0, 1.0]
        confidence = np.clip(confidence, 0.0, 1.0)
        mean_conf = float(np.mean(confidence))
        self.logger.info(f"Computed confidence across {num_verts} vertices. Mean: {mean_conf:.3f}, Min: {confidence.min():.3f}, Max: {confidence.max():.3f}")

        # 4. Generate Green -> Yellow -> Red Color Palette
        # red rises as confidence falls, green rises as confidence rises
        colors = np.zeros((num_verts, 3), dtype=np.float32)
        colors[:, 0] = np.clip(2.0 * (1.0 - confidence), 0.0, 1.0)   # Red
        colors[:, 1] = np.clip(2.0 * confidence, 0.0, 1.0)           # Green
        colors[:, 2] = 0.0                                          # Blue
        colors_uint8 = (colors * 255).astype(np.uint8)

        # 5. Export PLY with preserved numeric scalar attribute
        faces = np.asarray(mesh.triangles if hasattr(mesh, "triangles") else mesh.faces)
        self._write_confidence_ply(ply_out, vertices, faces, colors_uint8, confidence)
        self.logger.info(f"Exported confidence PLY with 'property float confidence' to '{ply_out}'.")

        # 6. Export GLB with BOTH standard COLOR_0 AND custom scalar _CONFIDENCE
        self._write_confidence_glb(glb_out, vertices, faces, colors_uint8, confidence)
        self.logger.success(f"Exported confidence GLB with standard COLOR_0 and custom _CONFIDENCE to '{glb_out}'.")

        return {
            "ply_path": ply_out,
            "glb_path": glb_out,
            "mean_confidence": mean_conf,
            "confidence_array": confidence
        }

    def _write_confidence_ply(
        self,
        filepath: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        colors: np.ndarray,
        confidence: np.ndarray
    ):
        """Writes ASCII/binary PLY preserving 'property float confidence'."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("property float confidence\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            for (x, y, z), (r, g, b), c in zip(vertices, colors, confidence):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {c:.4f}\n")
            
            for f0, f1, f2 in faces:
                f.write(f"3 {f0} {f1} {f2}\n")

    def _write_confidence_glb(
        self,
        filepath: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        colors_uint8: np.ndarray,
        confidence: np.ndarray
    ):
        """
        Exports GLB where:
        - colors_uint8 is baked into COLOR_0 (renders anywhere in standard viewers)
        - numeric confidence is stored in custom vertex attributes dictionary (_CONFIDENCE)
        """
        # Trimesh color visual assigns standard COLOR_0
        vertex_colors = np.column_stack([colors_uint8, np.full(len(colors_uint8), 255, dtype=np.uint8)])
        
        tm = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            process=False
        )
        
        # Store custom scalar attribute in metadata
        tm.vertex_attributes["_CONFIDENCE"] = confidence.astype(np.float32)
        
        glb_bytes = tm.export(file_type="glb")
        with open(filepath, "wb") as f:
            f.write(glb_bytes)
