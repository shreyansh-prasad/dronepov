"""
Scene data structures for the sih-face-completion pipeline.

Defines the canonical data containers passed between all pipeline stages.
No pipeline logic lives here — pure data + constants only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provenance constants  (AGENTS.md: never export without these)
# ---------------------------------------------------------------------------
class Provenance(IntEnum):
    OBSERVED = 0          # Face came directly from the reconstruction — fully trusted
    SYMMETRY_DERIVED = 1  # Face was mirrored from an observed counterpart — dimensionally accurate
    GENERATED = 2         # Face was synthesised by inpainting / fallback net — treat as estimate


PROVENANCE_COLORS: dict[Provenance, tuple[float, float, float]] = {
    Provenance.OBSERVED:         (0.35, 0.65, 0.35),  # green  — trusted
    Provenance.SYMMETRY_DERIVED: (0.40, 0.55, 0.80),  # blue   — derived but accurate
    Provenance.GENERATED:        (0.85, 0.50, 0.20),  # orange — synthetic, use with caution
}


# ---------------------------------------------------------------------------
# Semantic class registry  (must mirror upstream frame-selection schema.md)
# ---------------------------------------------------------------------------
class SemanticClass(IntEnum):
    BACKGROUND    = 0
    BUILDING      = 1
    ROAD          = 2
    TREE          = 3
    LOW_VEGETATION = 4
    STATIC_CAR    = 5
    MOVING_CAR    = 6
    HUMAN         = 7
    SKY           = 8
    WATER         = 9
    BARE_GROUND   = 10


# Which completion strategy each semantic class should use.
# "symmetry"  → PCA mirror (buildings, vehicles)
# "inpaint"   → OpenCV TELEA fallback (trees, humans — asymmetric / small)
# "none"      → skip (ground planes, sky, water — no meaningful occlusion)
COMPLETION_STRATEGY: dict[int, str] = {
    SemanticClass.BACKGROUND:     "none",
    SemanticClass.BUILDING:       "symmetry",
    SemanticClass.ROAD:           "none",
    SemanticClass.TREE:           "inpaint",
    SemanticClass.LOW_VEGETATION: "none",
    SemanticClass.STATIC_CAR:     "symmetry",
    SemanticClass.MOVING_CAR:     "symmetry",
    SemanticClass.HUMAN:          "inpaint",
    SemanticClass.SKY:            "none",
    SemanticClass.WATER:          "none",
    SemanticClass.BARE_GROUND:    "none",
}


# ---------------------------------------------------------------------------
# Core data containers
# ---------------------------------------------------------------------------
@dataclass
class CameraInfo:
    """Single COLMAP camera model."""
    camera_id: int
    model: str           # e.g. "PINHOLE", "RADIAL"
    width: int
    height: int
    params: np.ndarray   # [fx, fy, cx, cy, ...distortion]

    @property
    def K(self) -> np.ndarray:
        """Return 3×3 intrinsics matrix (assumes PINHOLE / SIMPLE_PINHOLE)."""
        fx, fy, cx, cy = self.params[:4]
        return np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=np.float64)


@dataclass
class FramePose:
    """COLMAP registered image: world-to-camera rotation + translation."""
    image_id: int
    image_name: str       # e.g. "frame_0042.png"
    camera_id: int
    R: np.ndarray         # (3, 3) rotation matrix — world→camera
    t: np.ndarray         # (3,)  translation — world→camera

    @property
    def world_to_cam(self) -> np.ndarray:
        """4×4 extrinsics matrix [R | t; 0 0 0 1]."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3,  3] = self.t
        return T

    @property
    def cam_to_world(self) -> np.ndarray:
        """
        Analytical inverse of world_to_cam (avoids full LU decomposition).
        For a rigid body transform [R|t], inverse = [R^T | -R^T t; 0 0 0 1].
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R.T
        T[:3,  3] = -self.R.T @ self.t
        return T

    @cached_property
    def centre_world(self) -> np.ndarray:
        """
        Camera optical centre in world coordinates (3,).
        Cached — called repeatedly by hole_detector and inpaint_completion.
        """
        return (-self.R.T @ self.t).astype(np.float64)


@dataclass
class ObjectInstance:
    """
    A single semantic object instance.

    Produced by the IO layer; consumed by detection, completion, and export.
    The mesh is always in world coordinates (same frame as the full scene mesh).
    """
    instance_id: int
    semantic_class: int           # SemanticClass value
    mesh: trimesh.Trimesh

    # Per-face provenance array — length == len(mesh.faces)
    # Initialised to OBSERVED; completion stages flip entries to SYMMETRY_DERIVED / GENERATED.
    provenance: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

    # Texture image (H×W×3 uint8) extracted from the COLMAP texture atlas, or None.
    texture_image: Optional[np.ndarray] = None

    # COLMAP camera poses that have line-of-sight to this instance.
    observing_cameras: list[FramePose] = field(default_factory=list)

    # Hole detection results — populated by Stage 2.
    # Boolean mask of length len(mesh.faces): True = face is a hole / missing.
    hole_mask: Optional[np.ndarray] = None

    # Symmetry detection results — populated by Stage 2.
    symmetry_plane: Optional[np.ndarray] = None   # (4,) plane equation [a,b,c,d] (ax+by+cz+d=0)
    symmetry_score: float = 0.0                   # ICP alignment score [0, 1]

    def __post_init__(self) -> None:
        # Always initialise provenance to OBSERVED for every face.
        # This is idempotent — if the caller already provided it, keep it.
        if self.mesh is not None and len(self.provenance) != len(self.mesh.faces):
            self.provenance = np.zeros(len(self.mesh.faces), dtype=np.int32)

    @property
    def semantic_name(self) -> str:
        try:
            return SemanticClass(self.semantic_class).name.lower()
        except ValueError:
            return f"class_{self.semantic_class}"

    @property
    def completion_strategy(self) -> str:
        return COMPLETION_STRATEGY.get(self.semantic_class, "none")

    @property
    def needs_completion(self) -> bool:
        return (
            self.completion_strategy != "none"
            and self.hole_mask is not None
            and self.hole_mask.any()
        )

    def log_summary(self) -> None:
        n_holes = int(self.hole_mask.sum()) if self.hole_mask is not None else -1
        logger.info(
            "[Instance %d | %s] faces=%d  holes=%d  strategy=%s  sym_score=%.3f",
            self.instance_id,
            self.semantic_name,
            len(self.mesh.faces),
            n_holes,
            self.completion_strategy,
            self.symmetry_score,
        )


@dataclass
class Scene:
    """
    Full reconstructed scene.

    Produced once by the IO layer; mutated in-place as completion stages run.
    """
    mesh: trimesh.Trimesh             # Full merged scene mesh (world coords)
    instances: list[ObjectInstance]   # Decomposed per-object instances
    cameras: dict[int, CameraInfo]    # camera_id → CameraInfo
    poses: list[FramePose]            # All registered image poses
    image_dir: Path                   # Directory containing source frame images
    scene_path: Path                  # Path to the source OBJ/PLY file

    @property
    def n_instances(self) -> int:
        return len(self.instances)

    def instances_for_completion(self) -> list[ObjectInstance]:
        """Return only instances that require and are ready for completion."""
        return [inst for inst in self.instances if inst.needs_completion]

    def log_summary(self) -> None:
        logger.info(
            "[Scene] source=%s  instances=%d  cameras=%d  poses=%d",
            self.scene_path.name,
            self.n_instances,
            len(self.cameras),
            len(self.poses),
        )
        for inst in self.instances:
            inst.log_summary()
