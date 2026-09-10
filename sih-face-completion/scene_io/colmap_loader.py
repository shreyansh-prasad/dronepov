"""
COLMAP scene loader.

Reads:
  - Dense textured mesh: scene.obj + scene.mtl  (or scene.ply)
  - COLMAP sparse model: cameras.txt, images.txt  (text format)
  - Upstream semantic masks: manifest.json from the frame-selection pipeline

Produces a fully populated Scene object ready for hole detection.

Expected directory layout (Team 2 handoff):
    <colmap_dir>/
        sparse/
            cameras.txt
            images.txt
        dense/
            fused.ply          ← dense point cloud (optional, used for UVs)
            meshed-poisson.obj ← textured mesh   ← PRIMARY INPUT
            meshed-poisson.mtl
            textures/
                *.png
        images/                ← source frames  (symlink OK)
    <manifest_path>            ← frame-selection pipeline output (manifest.json)
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from scipy.spatial import KDTree

from .scene import (
    CameraInfo,
    FramePose,
    ObjectInstance,
    Scene,
    SemanticClass,
)

logger = logging.getLogger(__name__)

# Minimum number of faces an instance must have to be processed.
_MIN_INSTANCE_FACES = 50

# COLMAP text-format parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_cameras_txt(path: Path) -> dict[int, CameraInfo]:
    """Parse COLMAP cameras.txt → {camera_id: CameraInfo}."""
    cameras: dict[int, CameraInfo] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id  = int(parts[0])
            model   = parts[1]
            width   = int(parts[2])
            height  = int(parts[3])
            params  = np.array([float(p) for p in parts[4:]], dtype=np.float64)
            cameras[cam_id] = CameraInfo(
                camera_id=cam_id,
                model=model,
                width=width,
                height=height,
                params=params,
            )
    logger.info("Loaded %d cameras from %s", len(cameras), path.name)
    return cameras


def _parse_images_txt(path: Path) -> list[FramePose]:
    """Parse COLMAP images.txt → list[FramePose].

    COLMAP text format: each registered image occupies two lines:
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID IMAGE_NAME
      POINTS2D[] ...
    """
    poses: list[FramePose] = []
    with path.open() as fh:
        lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        image_id   = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id  = int(parts[8])
        image_name = parts[9]

        R = _quat_to_rot(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)

        poses.append(FramePose(
            image_id=image_id,
            image_name=image_name,
            camera_id=camera_id,
            R=R,
            t=t,
        ))
        i += 2  # skip the POINTS2D line

    logger.info("Loaded %d image poses from %s", len(poses), path.name)
    return poses


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Quaternion → 3×3 rotation matrix (COLMAP convention: world-to-camera)."""
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)


# Mesh loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_mesh(colmap_dir: Path) -> trimesh.Trimesh:
    """
    Load the dense textured mesh produced by COLMAP/OpenMVS.

    Searches for meshed-poisson.obj first, then fused.ply as fallback.
    Raises FileNotFoundError if neither is found.
    """
    dense_dir = colmap_dir / "dense"
    candidates = [
        dense_dir / "meshed-poisson.obj",
        dense_dir / "scene.obj",
        dense_dir / "fused.obj",
        dense_dir / "fused.ply",
        dense_dir / "meshed-poisson.ply",
    ]
    for candidate in candidates:
        if candidate.exists():
            logger.info("Loading mesh from %s", candidate)
            scene_or_mesh = trimesh.load(str(candidate), force="mesh", process=False)
            if isinstance(scene_or_mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(
                    [g for g in scene_or_mesh.geometry.values()
                     if isinstance(g, trimesh.Trimesh)]
                )
            else:
                mesh = scene_or_mesh
            logger.info("Mesh loaded: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
            return mesh

    raise FileNotFoundError(
        f"No dense mesh found in {dense_dir}. "
        "Expected meshed-poisson.obj, scene.obj, fused.obj, or fused.ply. "
        "Run COLMAP dense reconstruction first."
    )


# Semantic instance segmentation using upstream pipeline output
# ─────────────────────────────────────────────────────────────────────────────

def _load_manifest(manifest_path: Path) -> list[dict]:
    """Load frame-selection pipeline manifest.json."""
    with manifest_path.open() as fh:
        data = json.load(fh)
    logger.info("Loaded manifest with %d frames from %s", len(data), manifest_path.name)
    return data


def _assign_face_semantics(
    mesh: trimesh.Trimesh,
    poses: list[FramePose],
    manifest: list[dict],
    image_dir: Path,
    cameras: dict[int, CameraInfo],
) -> np.ndarray:
    """
    Project per-frame semantic masks onto mesh faces.

    For each face:
      1. Compute face centroid in world space.
      2. Find the registered COLMAP frame whose camera is closest (most frontal).
      3. Project centroid into that frame's image plane.
      4. Sample the semantic mask at the projected pixel → face label.

    Returns a per-face label array shape (F,) dtype int32.
    """
    import cv2

    n_faces = len(mesh.faces)
    face_labels = np.zeros(n_faces, dtype=np.int32)   # default: BACKGROUND

    # Build {image_name: (pose, camera, mask_image)}
    frame_data: dict[str, tuple[FramePose, CameraInfo, np.ndarray]] = {}
    name_to_pose = {p.image_name: p for p in poses}

    for entry in manifest:
        img_name = Path(entry.get("frame_path", "")).name
        mask_rel = entry.get("mask_path", "")
        if not img_name or not mask_rel:
            continue
        pose = name_to_pose.get(img_name)
        if pose is None:
            continue
        cam = cameras.get(pose.camera_id)
        if cam is None:
            continue
        # Load mask if available
        mask_abs = Path(mask_rel) if Path(mask_rel).is_absolute() else Path.cwd() / mask_rel
        if not mask_abs.exists():
            continue
        mask_img = cv2.imread(str(mask_abs), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue
        frame_data[img_name] = (pose, cam, mask_img)

    if not frame_data:
        logger.warning(
            "No semantic masks could be loaded from manifest. "
            "All faces will be labelled BACKGROUND. "
            "Ensure mask_path entries in manifest.json point to valid files."
        )
        return face_labels

    # Face centroids in world space
    centroids = mesh.vertices[mesh.faces].mean(axis=1)   # (F, 3)
    face_normals = mesh.face_normals                      # (F, 3)

    frames = list(frame_data.values())
    n_frames = len(frames)
    logger.info("Projecting semantics using %d frames onto %d faces …", n_frames, n_faces)

    # For efficiency: batch project all centroids into each frame, pick best camera
    label_votes = np.zeros((n_faces, 11), dtype=np.int32)   # vote accumulator

    for pose, cam, mask_img in frames:
        K = cam.K
        T = pose.world_to_cam   # 4×4

        # Project centroids: cam_pts = R @ world_pts.T + t
        ones = np.ones((n_faces, 1))
        world_h = np.hstack([centroids, ones])          # (F, 4)
        cam_pts = (T @ world_h.T).T[:, :3]              # (F, 3)

        # Only keep faces in front of camera (positive z)
        in_front = cam_pts[:, 2] > 0.0

        # Project to image plane
        px = (K[0, 0] * cam_pts[:, 0] / cam_pts[:, 2] + K[0, 2])
        py = (K[1, 1] * cam_pts[:, 1] / cam_pts[:, 2] + K[1, 2])

        ih, iw = mask_img.shape
        valid = (
            in_front
            & (px >= 0) & (px < iw)
            & (py >= 0) & (py < ih)
        )

        xi = px[valid].astype(np.int32).clip(0, iw - 1)
        yi = py[valid].astype(np.int32).clip(0, ih - 1)
        labels = mask_img[yi, xi].astype(np.int32).clip(0, 10)

        valid_indices = np.where(valid)[0]
        np.add.at(label_votes, (valid_indices, labels), 1)

    # Majority-vote per face
    face_labels = label_votes.argmax(axis=1).astype(np.int32)
    unclassified = label_votes.max(axis=1) == 0
    face_labels[unclassified] = int(SemanticClass.BACKGROUND)

    unique, counts = np.unique(face_labels, return_counts=True)
    for cls, cnt in zip(unique, counts):
        name = SemanticClass(cls).name if cls in [c.value for c in SemanticClass] else str(cls)
        logger.info("  Class %-16s: %d faces (%.1f%%)", name, cnt, 100 * cnt / n_faces)

    return face_labels


def _extract_instances(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    poses: list[FramePose],
) -> list[ObjectInstance]:
    """
    Decompose the full scene mesh into per-class connected-component instances.

    Strategy:
      1. Group faces by semantic class.
      2. For each class, find connected components (shared edge adjacency).
      3. Each component with >= _MIN_INSTANCE_FACES becomes an ObjectInstance.
    """
    instances: list[ObjectInstance] = []
    instance_id = 0

    unique_classes = np.unique(face_labels)
    for cls in unique_classes:
        cls_face_indices = np.where(face_labels == cls)[0]
        if len(cls_face_indices) < _MIN_INSTANCE_FACES:
            continue

        # Extract sub-mesh for this class
        sub_mesh = mesh.submesh([cls_face_indices], append=True)
        if not isinstance(sub_mesh, trimesh.Trimesh):
            continue

        # Connected component decomposition within this class
        components = trimesh.graph.connected_component_labels(sub_mesh.face_adjacency)
        unique_comps = np.unique(components)

        for comp_id in unique_comps:
            comp_face_mask = components == comp_id
            comp_face_count = comp_face_mask.sum()
            if comp_face_count < _MIN_INSTANCE_FACES:
                continue

            comp_indices = np.where(comp_face_mask)[0]
            comp_mesh = sub_mesh.submesh([comp_indices], append=True)
            if not isinstance(comp_mesh, trimesh.Trimesh):
                continue

            # Find cameras that are roughly facing this instance
            instance_centre = comp_mesh.centroid
            observing: list[FramePose] = []
            for p in poses:
                # Camera-to-object vector in camera space: just check if z > 0 after transform
                pt_cam = p.R @ instance_centre + p.t
                if pt_cam[2] > 0:
                    observing.append(p)

            inst = ObjectInstance(
                instance_id=instance_id,
                semantic_class=int(cls),
                mesh=comp_mesh,
                observing_cameras=observing,
            )
            instances.append(inst)
            instance_id += 1
            logger.debug(
                "Instance %d: class=%s  faces=%d  observing_cameras=%d",
                instance_id - 1,
                inst.semantic_name,
                comp_face_count,
                len(observing),
            )

    logger.info("Extracted %d instances total.", len(instances))
    return instances


# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_colmap_scene(
    colmap_dir: str | Path,
    manifest_path: str | Path,
    image_dir: Optional[str | Path] = None,
) -> Scene:
    """
    Load a COLMAP-reconstructed scene and the upstream semantic manifest.

    Args:
        colmap_dir:    Root of the COLMAP reconstruction directory.
                       Must contain sparse/cameras.txt + images.txt and
                       dense/meshed-poisson.obj (or fused.ply).
        manifest_path: Path to manifest.json from the frame-selection pipeline.
        image_dir:     Directory of source frame images.
                       Defaults to <colmap_dir>/images/.

    Returns:
        Fully populated Scene object.
    """
    colmap_dir = Path(colmap_dir)
    manifest_path = Path(manifest_path)
    image_dir = Path(image_dir) if image_dir else colmap_dir / "images"

    sparse_dir = colmap_dir / "sparse"
    cameras_txt = sparse_dir / "cameras.txt"
    images_txt  = sparse_dir / "images.txt"

    for required in [cameras_txt, images_txt, manifest_path]:
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    cameras = _parse_cameras_txt(cameras_txt)
    poses   = _parse_images_txt(images_txt)
    mesh    = _load_mesh(colmap_dir)
    manifest = _load_manifest(manifest_path)

    face_labels = _assign_face_semantics(mesh, poses, manifest, image_dir, cameras)
    instances   = _extract_instances(mesh, face_labels, poses)

    scene = Scene(
        mesh=mesh,
        instances=instances,
        cameras=cameras,
        poses=poses,
        image_dir=image_dir,
        scene_path=colmap_dir,
    )
    scene.log_summary()
    return scene
