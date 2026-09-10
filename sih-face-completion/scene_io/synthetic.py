"""
Synthetic scene generator for offline testing.

Creates a realistic test scene without needing Team 2's COLMAP mesh.
Generates:
  - A partial building mesh (one full wall + roof; rear wall and side wall MISSING)
  - A partial car mesh (visible top + one side; opposite side MISSING)
  - Synthetic camera poses orbiting the scene

This lets every pipeline stage be validated end-to-end before the real data arrives.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

from .scene import (
    CameraInfo,
    FramePose,
    ObjectInstance,
    Scene,
    SemanticClass,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Mesh primitives
# ─────────────────────────────────────────────────────────────────────────────

def _box_with_hallucinated_back(width: float, height: float, depth: float) -> trimesh.Trimesh:
    """
    Create a closed box. The front/sides represent observed geometry,
    while the back faces represent hallucinated Poisson geometry.
    """
    w, h, d = width / 2, height / 2, depth / 2
    vertices = np.array([
        # front face (+Y)
        [-w, -h,  0], [ w, -h,  0], [ w,  h,  0], [-w,  h,  0],
        # top face (+Z)
        [-w, -h,  0], [ w, -h,  0], [ w, -h, height], [-w, -h, height],
        # left face (-X)
        [-w, -h,  0], [-w,  h,  0], [-w,  h, height], [-w, -h, height],
        # right face (+X)
        [ w, -h,  0], [ w,  h,  0], [ w,  h, height], [ w, -h, height],
        # roof (+Z top cap)
        [-w, -h, height], [ w, -h, height], [ w,  h, height], [-w,  h, height],
        # back face (-Y) - Hallucinated
        [-w,  h,  0], [ w,  h,  0], [ w,  h, height], [-w,  h, height],
        # bottom face (-Z)
        [-w, -h,  0], [ w, -h,  0], [ w,  h,  0], [-w,  h,  0],
    ], dtype=np.float64)
    faces = np.array([
        # front
        [0, 1, 2], [0, 2, 3],
        # top (front lower)
        [4, 6, 5], [4, 7, 6],
        # left
        [8, 9, 10], [8, 10, 11],
        # right
        [12, 14, 13], [12, 15, 14],
        # roof
        [16, 17, 18], [16, 18, 19],
        # back
        [20, 22, 21], [20, 23, 22],
        # bottom
        [24, 25, 26], [24, 26, 27],
    ], dtype=np.int32)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def _car_closed_shell(length: float = 4.0, width: float = 2.0, height: float = 1.5) -> trimesh.Trimesh:
    """
    Create a closed car shell. The unobserved side represents hallucinated geometry.
    """
    l, w, h = length / 2, width / 2, height
    vertices = np.array([
        # top face
        [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h],
        # front face
        [-l, -w, 0], [l, -w, 0], [l, -w, h], [-l, -w, h],
        # left face
        [-l, -w, 0], [-l, w, 0], [-l, w, h], [-l, -w, h],
        # right face
        [l, -w, 0], [l, w, 0], [l, w, h], [l, -w, h],
        # rear face
        [-l, w, 0], [l, w, 0], [l, w, h], [-l, w, h],
        # bottom
        [-l, -w, 0], [l, -w, 0], [l, w, 0], [-l, w, 0],
    ], dtype=np.float64)
    faces = np.array([
        # top
        [0, 1, 2], [0, 2, 3],
        # front
        [4, 6, 5], [4, 7, 6],
        # left
        [8, 9, 10], [8, 10, 11],
        # right
        [12, 13, 14], [12, 14, 15],
        # rear
        [16, 18, 17], [16, 19, 18],
        # bottom
        [20, 21, 22], [20, 22, 23],
    ], dtype=np.int32)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


# ─────────────────────────────────────────────────────────────────────────────
# Camera orbit generator
# ─────────────────────────────────────────────────────────────────────────────

def _orbit_cameras(
    centre: np.ndarray,
    radius: float,
    n_cameras: int,
    elevation_deg: float = 45.0,
    fov_deg: float = 60.0,
    image_width: int = 1920,
    image_height: int = 1080,
) -> tuple[dict[int, CameraInfo], list[FramePose]]:
    """
    Generate synthetic pinhole cameras orbiting `centre` at `radius`.

    Cameras face inward toward the scene centre, simulating a drone arc pass.
    Only the front half of the orbit is generated (simulating single-pass flight).
    """
    cameras: dict[int, CameraInfo] = {}
    poses:   list[FramePose] = []

    fx = fy = (image_width / 2) / np.tan(np.radians(fov_deg / 2))
    cx, cy = image_width / 2, image_height / 2

    cam_info = CameraInfo(
        camera_id=1,
        model="PINHOLE",
        width=image_width,
        height=image_height,
        params=np.array([fx, fy, cx, cy]),
    )
    cameras[1] = cam_info

    elev_rad = np.radians(elevation_deg)

    # Only front 180° arc — this is what a single drone pass gives you.
    # The rear 180° is the "unseen side" we need to complete.
    angles = np.linspace(-np.pi / 2, np.pi / 2, n_cameras)

    for i, angle in enumerate(angles):
        # Camera position on the orbit
        cam_pos = centre + radius * np.array([
            np.cos(angle) * np.cos(elev_rad),
            np.sin(angle) * np.cos(elev_rad),
            np.sin(elev_rad),
        ])

        # Look toward scene centre: compute R, t
        forward = centre - cam_pos
        forward /= np.linalg.norm(forward)

        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        # Rotation matrix (camera axes as rows: right, -up, forward)
        R = np.stack([right, -up, forward], axis=0)   # (3,3)
        t = -R @ cam_pos                               # world-to-camera translation

        poses.append(FramePose(
            image_id=i + 1,
            image_name=f"frame_{i:04d}.png",
            camera_id=1,
            R=R,
            t=t,
        ))

    return cameras, poses


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_scene(output_dir: str | Path = ".") -> Scene:
    """
    Generate a minimal synthetic scene for end-to-end pipeline testing.

    Places:
      - 1 partial building (rear wall missing) at origin
      - 1 partial car (right side missing) 8 m to the right
    Generates 12 synthetic cameras covering the front-hemisphere only.

    Args:
        output_dir: Where to write synthetic mesh files for inspection.

    Returns:
        A Scene object ready to be passed to the detection and completion stages.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build partial building ────────────────────────────────────────────
    building_mesh = _box_with_hallucinated_back(width=10.0, height=8.0, depth=10.0)
    building_mesh.apply_translation([0.0, 0.0, 0.0])

    building_inst = ObjectInstance(
        instance_id=0,
        semantic_class=int(SemanticClass.BUILDING),
        mesh=building_mesh,
    )

    # ── Build partial car ─────────────────────────────────────────────────
    car_mesh = _car_closed_shell(length=4.0, width=2.0, height=1.5)
    car_mesh.apply_translation([8.0, 0.0, 0.0])

    car_inst = ObjectInstance(
        instance_id=1,
        semantic_class=int(SemanticClass.STATIC_CAR),
        mesh=car_mesh,
    )

    # ── Full scene mesh (merge both) ──────────────────────────────────────
    full_mesh = trimesh.util.concatenate([building_mesh, car_mesh])

    # ── Cameras ───────────────────────────────────────────────────────────
    scene_centre = full_mesh.centroid
    cameras, poses = _orbit_cameras(
        centre=scene_centre,
        radius=25.0,
        n_cameras=12,
        elevation_deg=40.0,
    )

    # Assign observing cameras to instances (all front-arc cameras see both)
    building_inst.observing_cameras = poses
    car_inst.observing_cameras = poses

    # Save synthetic meshes for visual inspection
    building_mesh.export(str(output_dir / "synthetic_building_partial.obj"))
    car_mesh.export(str(output_dir / "synthetic_car_partial.obj"))
    full_mesh.export(str(output_dir / "synthetic_scene_partial.obj"))

    logger.info(
        "Synthetic scene created: building(%d faces) + car(%d faces) | %d cameras",
        len(building_mesh.faces),
        len(car_mesh.faces),
        len(poses),
    )

    return Scene(
        mesh=full_mesh,
        instances=[building_inst, car_inst],
        cameras=cameras,
        poses=poses,
        image_dir=output_dir,
        scene_path=output_dir / "synthetic_scene_partial.obj",
    )
