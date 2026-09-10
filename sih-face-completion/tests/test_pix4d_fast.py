import open3d as o3d
import numpy as np
from pathlib import Path

def run_poisson_on_fbx():
    fbx_path = r"C:\Projects\sih\Building-3d_mesh.fbx"
    out_dir = Path("tests_output/pix4d_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / "scene_completed.obj"
    
    print(f"Loading {fbx_path}...")
    mesh = o3d.io.read_triangle_mesh(fbx_path)
    print(f"Loaded mesh with {len(mesh.vertices)} vertices.")
    
    # Ensure normals exist for Poisson
    if not mesh.has_vertex_normals():
        print("Computing normals...")
        mesh.compute_vertex_normals()

    print("Running Poisson Surface Reconstruction (this acts as our hole filler)...")
    # Convert mesh to point cloud for Poisson
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals
    
    # Apply Poisson Surface Reconstruction
    # Depth 9 gives great detail, depth 10 is very dense.
    mesh_out, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    # We want to crop the reconstructed mesh to the original bounding box 
    # to remove any low-density extrapolated bubbles.
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh_out = mesh_out.crop(bbox)

    print(f"Exporting filled mesh with {len(mesh_out.triangles)} triangles...")
    o3d.io.write_triangle_mesh(str(obj_path), mesh_out)
    print("Done! You can now view the completed mesh.")

if __name__ == "__main__":
    run_poisson_on_fbx()
