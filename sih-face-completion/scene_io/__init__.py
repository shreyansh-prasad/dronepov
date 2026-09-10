"""io package — scene loading utilities."""
from .scene import (
    Scene,
    ObjectInstance,
    CameraInfo,
    FramePose,
    Provenance,
    SemanticClass,
    COMPLETION_STRATEGY,
    PROVENANCE_COLORS,
)
from .colmap_loader import load_colmap_scene
from .synthetic import make_synthetic_scene

__all__ = [
    "Scene",
    "ObjectInstance",
    "CameraInfo",
    "FramePose",
    "Provenance",
    "SemanticClass",
    "COMPLETION_STRATEGY",
    "PROVENANCE_COLORS",
    "load_colmap_scene",
    "make_synthetic_scene",
]
