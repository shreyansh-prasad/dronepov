import sys
import os
from pathlib import Path
import trimesh
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scene_io.scene import ObjectInstance, Scene, SemanticClass
from output import SceneExporter
from scene_io.synthetic import _orbit_cameras

def main():
    obj_path = r"tests_output\pix4d_test\scene_completed.obj"
    out_dir = Path("tests_output/pix4d_test")
    
    print(f"Loading {obj_path}...")
    mesh = trimesh.load(obj_path)
    
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.geometry[list(mesh.geometry.keys())[0]]

    # Ensure mesh has extents
    print(f"Mesh has {len(mesh.faces)} faces. Extents: {mesh.extents}")

    # Generate cameras
    cameras, poses = _orbit_cameras(
        centre=mesh.centroid,
        radius=max(mesh.extents) * 1.5,
        n_cameras=30,
        elevation_deg=30.0
    )
    
    inst = ObjectInstance(instance_id=0, semantic_class=0, mesh=mesh)
    scene = Scene(
        mesh=mesh,
        instances=[inst],
        cameras=cameras,
        poses=poses,
        image_dir=out_dir,
        scene_path=out_dir / "scene_completed.obj"
    )

    exporter = SceneExporter(out_dir)
    print("Rendering turntable...")
    exporter._render_turntable(scene, out_dir / "turntable_pix4d.mp4")
    print("Done rendering turntable!")

if __name__ == "__main__":
    main()
