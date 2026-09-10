"""
Symmetry plane detector for 3D object instances.

Algorithm (first principles):
  1. PCA on the point cloud → 3 candidate symmetry planes (each perpendicular
     to a principal axis, passing through the centroid).
  2. For each candidate plane, mirror the point cloud across it.
  3. Run ICP (Iterative Closest Point) between the original and mirrored cloud.
  4. The alignment score = 1 / (1 + mean_residual_distance). Range [0, 1].
  5. Accept the best plane if score > SYMMETRY_ACCEPT_THRESHOLD.

Why PCA + ICP and not just point-cloud mirroring?
  - PCA gives 3 physically meaningful candidate planes quickly (O(N) after cov).
  - ICP verifies how well the mirrored cloud aligns to the original,
    which is the ONLY reliable way to detect false-positive symmetry.
  - Testing all 3 PCA planes guarantees we find the correct axis even for
    oblique objects (e.g. a building placed diagonally in the scene).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import open3d as o3d

from scene_io.scene import ObjectInstance

logger = logging.getLogger(__name__)


@dataclass
class SymmetryDetectorConfig:
    symmetry_accept_threshold: float = 0.60   # ICP score below this → no symmetry found
    max_icp_iterations: int = 50
    icp_max_correspondence_dist_ratio: float = 0.03  # fraction of AABB diagonal (was 0.10 — too large)
    n_sample_points: int = 2048               # points sampled from mesh for ICP


class SymmetryDetector:
    """
    Detects the dominant bilateral symmetry plane of an ObjectInstance.

    Usage:
        detector = SymmetryDetector(config)
        detector.detect(instance)
        # instance.symmetry_plane and instance.symmetry_score are now populated
    """

    def __init__(self, config: SymmetryDetectorConfig | None = None) -> None:
        self.config = config or SymmetryDetectorConfig()

    def detect(self, instance: ObjectInstance) -> None:
        """
        Populate instance.symmetry_plane and instance.symmetry_score in-place.

        If no reliable symmetry is found (score < threshold), symmetry_plane
        is set to None and the completion dispatcher will fall through to inpainting.
        """
        cfg = self.config
        mesh = instance.mesh

        if len(mesh.vertices) < 4:
            logger.warning(
                "[Instance %d] Too few vertices for symmetry detection.", instance.instance_id
            )
            instance.symmetry_plane = None
            instance.symmetry_score = 0.0
            return

        # Sample points from the mesh surface
        pts = self._sample_points(mesh, cfg.n_sample_points)
        if len(pts) < 6:
            instance.symmetry_plane = None
            instance.symmetry_score = 0.0
            return

        centroid = pts.mean(axis=0)

        # Use Oriented Bounding Box (OBB) to extract 3 principal axes
        # OBB is much more robust to missing internal geometry than raw PCA
        try:
            obb = mesh.bounding_box_oriented
            transform = obb.primitive.transform
            # The top-left 3x3 of the transform matrix are the principal axes
            axes = transform[:3, :3].T
        except Exception as exc:
            logger.warning("[Instance %d] OBB failed: %s", instance.instance_id, exc)
            instance.symmetry_plane = None
            instance.symmetry_score = 0.0
            return

        aabb_diag = float(np.linalg.norm(mesh.bounding_box.extents))
        max_corr = aabb_diag * cfg.icp_max_correspondence_dist_ratio

        best_plane: np.ndarray | None = None
        best_score: float = 0.0

        for axis in axes:
            plane = self._make_plane(centroid, axis)
            mirrored = self._mirror_points(pts, plane)
            score = self._icp_score(pts, mirrored, max_corr, cfg.max_icp_iterations)

            logger.debug(
                "[Instance %d] Axis %s → score %.4f",
                instance.instance_id,
                np.round(axis, 2),
                score,
            )

            if score > best_score:
                best_score = score
                best_plane = plane

        if best_score >= cfg.symmetry_accept_threshold:
            instance.symmetry_plane = best_plane
            instance.symmetry_score = best_score
            logger.info(
                "[Instance %d | %s] Symmetry accepted: score=%.3f  plane=%s",
                instance.instance_id,
                instance.semantic_name,
                best_score,
                np.round(best_plane, 3) if best_plane is not None else None,
            )
        else:
            instance.symmetry_plane = None
            instance.symmetry_score = best_score
            logger.info(
                "[Instance %d | %s] Symmetry REJECTED: best score=%.3f < threshold=%.2f",
                instance.instance_id,
                instance.semantic_name,
                best_score,
                cfg.symmetry_accept_threshold,
            )

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sample_points(mesh, n: int) -> np.ndarray:
        """Sample n points from mesh surface. Falls back to vertices if sampling fails."""
        import trimesh
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, n)
            return pts.astype(np.float64)
        except Exception:
            return mesh.vertices[:n].astype(np.float64)

    @staticmethod
    def _make_plane(point: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Return plane equation [a, b, c, d] where ax+by+cz+d=0."""
        n = normal / (np.linalg.norm(normal) + 1e-12)
        d = -float(n @ point)
        return np.array([n[0], n[1], n[2], d], dtype=np.float64)

    @staticmethod
    def _mirror_points(pts: np.ndarray, plane: np.ndarray) -> np.ndarray:
        """
        Mirror all points across the plane ax+by+cz+d=0.

        Reflection formula: p' = p - 2*(a·px + b·py + c·pz + d) / (a²+b²+c²) * n
        """
        a, b, c, d = plane
        n = np.array([a, b, c], dtype=np.float64)
        n_sq = float(n @ n)
        if n_sq < 1e-12:
            return pts.copy()
        dist = (pts @ n + d) / n_sq          # (N,)
        return pts - 2.0 * dist[:, None] * n  # (N, 3)

    @staticmethod
    def _icp_score(
        source: np.ndarray,
        target: np.ndarray,
        max_correspondence_dist: float,
        max_iter: int,
    ) -> float:
        """
        Run Open3D point-to-point ICP and return a score in [0, 1].

        Score = 1 / (1 + mean_residual). Higher = better alignment.
        """
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(source)

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(target)

        try:
            result = o3d.pipelines.registration.registration_icp(
                src_pcd,
                tgt_pcd,
                max_correspondence_distance=max_correspondence_dist,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=max_iter
                ),
            )
            rmse = result.inlier_rmse
        except Exception as exc:
            logger.debug("ICP failed: %s", exc)
            return 0.0

        if rmse == 0.0:
            return 1.0
        return float(1.0 / (1.0 + rmse))
