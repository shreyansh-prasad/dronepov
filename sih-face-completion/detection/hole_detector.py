"""
Hole / no-data detector using Open3D raycasting.

Algorithm (first principles):
  1. Build a RaycastingScene from the instance mesh.
  2. Cast rays from N uniformly distributed viewpoints on a hemisphere above the mesh.
  3. A face that is NEVER hit by any ray = it faces away from all observed cameras
     = it is on the "unseen" side = it is a hole / missing region.
  4. Small isolated un-hit patches (< min_hole_area_ratio) are ignored (likely occluded
     by geometry, not genuinely absent).
  5. Returns a boolean hole_mask per face: True = missing / to be completed.

Why raycasting and NOT a simple backface-cull?
  - Backface-cull only works if the mesh is already closed and watertight.
    COLMAP dense meshes are never perfectly watertight.
  - Raycasting works on open shells and correctly identifies internal vs external faces.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import trimesh

from scene_io.scene import ObjectInstance

logger = logging.getLogger(__name__)


@dataclass
class HoleDetectorConfig:
    """Tunable parameters — use a dataclass, never raw dicts (AGENTS.md)."""
    hit_threshold: int = 1             # A face must be seen from >= this many cameras to be OBSERVED
    min_hole_area_ratio: float = 0.005 # Holes < 0.5% of total face count are noise (increased precision)
    max_correspondence_dist: float = 0.5  # metres — ICP correspondence dist for symmetry (separate)

    # Legacy fields kept for config-file compatibility; ignored internally.
    n_viewpoints: int = 64
    n_rays_per_viewpoint: int = 512
    hemisphere_scale: float = 2.5


class HoleDetector:
    """
    Detects missing / occluded faces in a partial mesh via hemisphere raycasting.

    Usage:
        detector = HoleDetector(config)
        detector.detect(instance)
        # instance.hole_mask is now populated
    """

    def __init__(self, config: HoleDetectorConfig | None = None) -> None:
        self.config = config or HoleDetectorConfig()

    def detect(self, instance: ObjectInstance) -> None:
        """
        Populate instance.hole_mask in-place.

        Modifies:
            instance.hole_mask (np.ndarray bool, shape (F,))
        """
        mesh = instance.mesh
        if len(mesh.faces) == 0:
            logger.warning("[Instance %d] Empty mesh, skipping hole detection.", instance.instance_id)
            instance.hole_mask = np.zeros(0, dtype=bool)
            return

        cfg = self.config
        n_faces = len(mesh.faces)

        if not instance.observing_cameras:
            logger.warning("[Instance %d] No observing cameras. All faces marked as holes.", instance.instance_id)
            instance.hole_mask = np.ones(n_faces, dtype=bool)
            return

        # Build Open3D raycasting scene
        o3d_mesh = self._to_o3d(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d_mesh)

        face_centers = mesh.triangles_center
        face_normals = mesh.face_normals
        face_hit_count = np.zeros(n_faces, dtype=np.int32)

        for pose in instance.observing_cameras:
            cam_center = pose.centre_world.astype(np.float32)

            cam_to_face = face_centers - cam_center
            norms = np.linalg.norm(cam_to_face, axis=1, keepdims=True)
            norms = np.where(norms < 1e-9, 1.0, norms)
            dirs = (cam_to_face / norms).astype(np.float32)

            # Face points towards camera if dot(normal, dir) < 0
            dots = np.sum(dirs * face_normals, axis=1)
            front_facing = dots < -1e-3  # slight margin

            target_faces = np.where(front_facing)[0]
            if len(target_faces) == 0:
                continue

            rays_np = np.zeros((len(target_faces), 6), dtype=np.float32)
            rays_np[:, :3] = cam_center
            rays_np[:, 3:] = dirs[target_faces]

            rays = o3d.core.Tensor(rays_np, dtype=o3d.core.float32)
            ans = scene.cast_rays(rays)

            hit_ids = ans["primitive_ids"].numpy()   # which triangle was hit first
            t_hit   = ans["t_hit"].numpy()            # distance to the first hit

            # A face is "directly visible" if:
            #   (a) the first hit IS the target face (not blocked by a closer one), OR
            #   (b) t_hit is within a small tolerance of the true distance.
            # We use (a) as the primary test, with a fallback t_hit guard.
            true_dist = np.linalg.norm(
                face_centers[target_faces] - cam_center, axis=1
            ).astype(np.float32)
            # Allow 2 cm slack for floating-point near-surface hits
            directly_visible = (hit_ids == target_faces) | (np.abs(t_hit - true_dist) < 0.02)

            hit_faces = target_faces[directly_visible]
            hit_faces = hit_faces[hit_faces < n_faces]
            np.add.at(face_hit_count, hit_faces, 1)

        # Faces with fewer hits than threshold are candidates for holes
        raw_hole_mask = face_hit_count < cfg.hit_threshold

        # Filter out tiny isolated patches (noise / micro-occluded faces)
        hole_mask = self._filter_small_patches(mesh, raw_hole_mask, cfg.min_hole_area_ratio)

        instance.hole_mask = hole_mask
        n_holes = int(hole_mask.sum())
        logger.info(
            "[Instance %d | %s] Hole detection: %d / %d faces are holes (%.1f%%)",
            instance.instance_id,
            instance.semantic_name,
            n_holes,
            n_faces,
            100.0 * n_holes / max(n_faces, 1),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_o3d(mesh: trimesh.Trimesh) -> o3d.t.geometry.TriangleMesh:
        """Convert trimesh → Open3D tensor TriangleMesh."""
        o3d_mesh = o3d.t.geometry.TriangleMesh()
        o3d_mesh.vertex.positions = o3d.core.Tensor(
            mesh.vertices.astype(np.float32), dtype=o3d.core.float32
        )
        o3d_mesh.triangle.indices = o3d.core.Tensor(
            mesh.faces.astype(np.int32), dtype=o3d.core.int32
        )
        return o3d_mesh



    @staticmethod
    def _filter_small_patches(
        mesh: trimesh.Trimesh,
        raw_mask: np.ndarray,
        min_ratio: float,
    ) -> np.ndarray:
        """
        Remove tiny isolated hole patches that are likely noise.

        Uses face adjacency graph connected components:
        any hole component with < min_ratio of total faces is cleared.
        """
        if not raw_mask.any():
            return raw_mask.copy()

        # Build adjacency only for hole faces
        n_faces = len(mesh.faces)
        hole_indices = np.where(raw_mask)[0]

        # Face adjacency: each row is (face_a, face_b) sharing an edge
        adj = mesh.face_adjacency                     # (E, 2)
        # Keep only edges where BOTH faces are holes
        both_hole = raw_mask[adj[:, 0]] & raw_mask[adj[:, 1]]
        hole_adj = adj[both_hole]

        # Union-Find on hole faces
        parent = {i: i for i in hole_indices}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for a, b in hole_adj:
            union(int(a), int(b))

        # Count component sizes
        from collections import Counter
        roots = {i: find(i) for i in hole_indices}
        comp_sizes = Counter(roots.values())

        min_size = max(1, int(n_faces * min_ratio))
        filtered = np.zeros(n_faces, dtype=bool)
        for face_idx in hole_indices:
            root = roots[face_idx]
            if comp_sizes[root] >= min_size:
                filtered[face_idx] = True

        n_removed = int(raw_mask.sum()) - int(filtered.sum())
        if n_removed > 0:
            logger.debug("Filtered %d micro-hole faces (noise).", n_removed)

        return filtered
