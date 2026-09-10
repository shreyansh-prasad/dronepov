"""
Symmetry-based face completion (PATH A) using Poisson Surface Reconstruction.

Completes the missing faces of a mesh by mirroring the observed geometry
across the detected symmetry plane, then running Open3D's Poisson Surface 
Reconstruction to generate a perfectly watertight, seamlessly colored mesh.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import trimesh
import open3d as o3d

from scene_io.scene import ObjectInstance, Provenance

logger = logging.getLogger(__name__)


class SymmetryCompletion:
    """
    Fills holes by reflecting observed geometry across the symmetry plane
    and using Poisson Reconstruction for a watertight result.
    """

    def complete(self, instance: ObjectInstance) -> bool:
        """
        Perform in-place Poisson symmetry completion.

        Returns True if successful, False otherwise.
        Modifies instance.mesh and instance.provenance in-place.
        """
        if instance.symmetry_plane is None:
            logger.warning(
                "[Instance %d] No symmetry plane — cannot use symmetry completion.",
                instance.instance_id,
            )
            return False

        if instance.hole_mask is None or not instance.hole_mask.any():
            logger.info(
                "[Instance %d] No holes detected — nothing to complete.",
                instance.instance_id,
            )
            return False

        plane = instance.symmetry_plane
        mesh = instance.mesh
        hole_mask = instance.hole_mask

        n_holes = int(hole_mask.sum())
        logger.info(
            "[Instance %d | %s] Symmetry completion: mirroring %d hole faces.",
            instance.instance_id, instance.semantic_name, n_holes,
        )

        # 1. Extract observed points, normals, and colors
        observed_face_indices = np.where(~hole_mask)[0]
        observed_faces = mesh.faces[observed_face_indices]
        used_vertex_ids = np.unique(observed_faces.ravel())

        pts = mesh.vertices[used_vertex_ids]
        
        # Check if normals exist, otherwise compute them
        if mesh.vertex_normals is None or len(mesh.vertex_normals) != len(mesh.vertices):
            mesh.fix_normals()
        normals = mesh.vertex_normals[used_vertex_ids]

        has_colors = mesh.visual is not None and hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) == len(mesh.vertices)
        if has_colors:
            colors = np.array(mesh.visual.vertex_colors)[used_vertex_ids, :3] / 255.0
        else:
            colors = np.ones((len(pts), 3)) * 0.8

        # 2. Mirror points and normals
        mirrored_pts = self._mirror_points(pts, plane)
        mirrored_normals = self._mirror_vectors(normals, plane)
        mirrored_colors = colors.copy()

        # 3. Combine into a dense point cloud
        all_pts = np.vstack([pts, mirrored_pts])
        all_normals = np.vstack([normals, mirrored_normals])
        all_colors = np.vstack([colors, mirrored_colors])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        pcd.normals = o3d.utility.Vector3dVector(all_normals)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)

        # 4. Poisson Surface Reconstruction
        logger.info("[Instance %d] Running Poisson Surface Reconstruction...", instance.instance_id)
        try:
            poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        except Exception as exc:
            logger.error("[Instance %d] Poisson failed: %s", instance.instance_id, exc)
            return False
        
        # 5. Crop extrapolated geometry
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox.scale(1.1, bbox.get_center())
        poisson_mesh = poisson_mesh.crop(bbox)
        
        if len(poisson_mesh.triangles) == 0:
            logger.warning("[Instance %d] Poisson mesh is empty after cropping.", instance.instance_id)
            return False

        # Convert back to trimesh
        new_vertices = np.asarray(poisson_mesh.vertices)
        new_faces = np.asarray(poisson_mesh.triangles)
        new_colors = np.asarray(poisson_mesh.vertex_colors)
        
        vertex_colors = (np.clip(new_colors, 0.0, 1.0) * 255).astype(np.uint8)
        vertex_colors = np.hstack((vertex_colors, np.full((len(vertex_colors), 1), 255, dtype=np.uint8)))

        completed_mesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            vertex_colors=vertex_colors,
            process=True,
        )

        n_orig_faces = len(mesh.faces)
        n_final_faces = len(completed_mesh.faces)

        provenance = np.full(n_final_faces, Provenance.SYMMETRY_DERIVED, dtype=np.int32)

        instance.mesh = completed_mesh
        instance.provenance = provenance

        logger.info(
            "[Instance %d] Poisson completion done: %d → %d faces.",
            instance.instance_id,
            n_orig_faces,
            n_final_faces,
        )
        return True

    @staticmethod
    def _mirror_points(pts: np.ndarray, plane: np.ndarray) -> np.ndarray:
        a, b, c, d = plane
        n = np.array([a, b, c], dtype=np.float64)
        n_sq = float(n @ n)
        if n_sq < 1e-12:
            return pts.copy()
        dist = (pts @ n + d) / n_sq
        return pts - 2.0 * dist[:, None] * n

    @staticmethod
    def _mirror_vectors(vecs: np.ndarray, plane: np.ndarray) -> np.ndarray:
        a, b, c, d = plane
        n = np.array([a, b, c], dtype=np.float64)
        n_sq = float(n @ n)
        if n_sq < 1e-12:
            return vecs.copy()
        dist = (vecs @ n) / n_sq
        return vecs - 2.0 * dist[:, None] * n
