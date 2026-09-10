"""
Inpainting-based face completion fallback (PATH C) using Poisson Reconstruction.

Used for asymmetric objects (trees, humans, irregular structures)
where symmetry cannot be trusted.

Algorithm:
  1. Extract observed vertices, normals, and colors into a dense point cloud.
  2. Do not mirror.
  3. Run Open3D's Poisson Surface Reconstruction directly on the observed points.
  4. Poisson naturally extrapolates and closes the mesh seamlessly based on 
     boundary normals, acting as a 3D structural inpainter.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import trimesh
import open3d as o3d

from scene_io.scene import ObjectInstance, Provenance

logger = logging.getLogger(__name__)


class InpaintCompletion:
    """
    Fills holes in asymmetric objects via Poisson Surface Reconstruction.
    """

    def complete(self, instance: ObjectInstance) -> bool:
        """
        Perform in-place asymmetric Poisson completion.

        Returns True if successful, False otherwise.
        Modifies instance.mesh and instance.provenance in-place.
        """
        if instance.hole_mask is None or not instance.hole_mask.any():
            logger.info("[Instance %d] No holes — skipping asymmetric completion.", instance.instance_id)
            return False

        mesh = instance.mesh
        n_holes = int(instance.hole_mask.sum())
        logger.info(
            "[Instance %d | %s] Asymmetric completion: processing %d hole faces.",
            instance.instance_id, instance.semantic_name, n_holes,
        )

        # 1. Extract observed points, normals, and colors
        observed_face_indices = np.where(~instance.hole_mask)[0]
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

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 2. Poisson Surface Reconstruction (extrapolates unobserved areas natively)
        logger.info("[Instance %d] Running Poisson Surface Reconstruction (Asymmetric)...", instance.instance_id)
        try:
            poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        except Exception as exc:
            logger.error("[Instance %d] Poisson failed: %s", instance.instance_id, exc)
            return False
        
        # 3. Crop extrapolated geometry
        bbox = pcd.get_axis_aligned_bounding_box()
        # For asymmetric shapes, we allow slightly more extrapolation (bubble effect)
        bbox.scale(1.2, bbox.get_center())
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

        # Tag all faces as GENERATED since it's an asymmetric mathematical hallucination
        provenance = np.full(n_final_faces, Provenance.GENERATED, dtype=np.int32)

        instance.mesh = completed_mesh
        instance.provenance = provenance

        logger.info(
            "[Instance %d] Asymmetric Poisson completion done: %d → %d faces.",
            instance.instance_id,
            n_orig_faces,
            n_final_faces,
        )
        return True
