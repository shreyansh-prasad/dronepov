import sys
import os
from pathlib import Path
import trimesh
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scene_io.scene import ObjectInstance, Scene, SemanticClass
from detection import HoleDetector, HoleDetectorConfig, SymmetryDetector, SymmetryDetectorConfig
from completion import CompletionDispatcher
from output import SceneExporter
from scene_io.synthetic import _orbit_cameras

def main():
    fbx_path = r"C:\Projects\sih\Building-3d_mesh.fbx"
    out_dir = Path("tests_output/pix4d_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {fbx_path}...")
    # force load as single mesh if possible
    mesh_or_scene = trimesh.load(fbx_path, force='mesh')
    
    if isinstance(mesh_or_scene, trimesh.Scene):
        print("Loaded as Scene, concatenating meshes...")
        mesh = trimesh.util.concatenate([geom for geom in mesh_or_scene.geometry.values()])
    else:
        mesh = mesh_or_scene
        
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
    
    # Let's save a copy as OBJ just in case Trimesh parsed it weirdly
    mesh.export(out_dir / "input_converted.obj")
    
    # 2. Setup Scene structure for our pipeline
    inst = ObjectInstance(
        instance_id=0,
        semantic_class=int(SemanticClass.BUILDING),
        mesh=mesh
    )
    
    # Generate orbital cameras around the mesh to simulate drone capture
    print("Generating simulated drone cameras...")
    extents = mesh.extents
    radius = max(extents) * 1.5
    cameras, poses = _orbit_cameras(
        centre=mesh.centroid,
        radius=radius,
        n_cameras=36,
        elevation_deg=30.0
    )
    inst.observing_cameras = poses
    
    scene = Scene(
        mesh=mesh,
        instances=[inst],
        cameras=cameras,
        poses=poses,
        image_dir=out_dir,
        scene_path=out_dir / "input_converted.obj"
    )

    # 3. Detect Holes
    print("Detecting holes...")
    HoleDetector(HoleDetectorConfig()).detect(inst)
    print(f"Detected holes: {inst.hole_faces.shape if inst.hole_faces is not None else 'None'}")
    
    # 4. Detect Symmetry (for mirroring or Poisson fallback)
    print("Detecting symmetry...")
    SymmetryDetector(SymmetryDetectorConfig(symmetry_accept_threshold=0.3)).detect(inst)
    print(f"Is symmetric: {inst.is_symmetric}")
    
    # 5. Run Completion
    print("Running Completion Dispatcher...")
    CompletionDispatcher().run([inst])
    
    # 6. Export results
    print("Exporting results...")
    exporter = SceneExporter(out_dir)
    exporter.export(scene)
    print(f"Success! Check the output at {out_dir}")

if __name__ == "__main__":
    main()
