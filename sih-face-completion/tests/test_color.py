import sys
from pathlib import Path
import trimesh
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scene_io.scene import ObjectInstance, Scene, SemanticClass
from detection import HoleDetector, SymmetryDetector, SymmetryDetectorConfig
from completion import CompletionDispatcher
from output import SceneExporter
from scene_io.synthetic import _orbit_cameras

def main():
    # 1. Create a colored sphere
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    
    # Assign colors: top half red, bottom half blue
    colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    for i, v in enumerate(mesh.vertices):
        if v[2] > 0:
            colors[i] = [255, 0, 0, 255] # Red
        else:
            colors[i] = [0, 0, 255, 255] # Blue
    mesh.visual.vertex_colors = colors

    # 2. Punch a hole in the middle (near the equator where red and blue meet)
    centers = mesh.triangles.mean(axis=1)
    # Remove faces in the front
    keep_mask = (centers[:, 0] < 5.0) | (np.abs(centers[:, 2]) > 3.0)
    mesh.update_faces(keep_mask)
    mesh.remove_unreferenced_vertices()

    out_dir = Path("tests_output/color_test")
    out_dir.mkdir(exist_ok=True, parents=True)
    mesh.export(out_dir / "input_colored_hole.obj")

    # 3. Setup Scene
    inst = ObjectInstance(
        instance_id=1,
        semantic_class=int(SemanticClass.BUILDING),
        mesh=mesh
    )
    
    cameras, poses = _orbit_cameras(
        centre=mesh.centroid,
        radius=30.0,
        n_cameras=10,
        elevation_deg=20.0
    )
    inst.observing_cameras = poses
    
    scene = Scene(
        mesh=mesh,
        instances=[inst],
        cameras=cameras,
        poses=poses,
        image_dir=out_dir,
        scene_path=out_dir / "input_colored_hole.obj"
    )

    # 4. Run Pipeline
    HoleDetector().detect(inst)
    SymmetryDetector(SymmetryDetectorConfig()).detect(inst)
    CompletionDispatcher().run([inst])

    # 5. Export
    exporter = SceneExporter(out_dir)
    exporter.export(scene)
    
    print("Done! Check tests_output/color_test for the results.")

if __name__ == "__main__":
    main()
