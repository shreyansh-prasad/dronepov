import open3d as o3d
import trimesh
import numpy as np

def convert_and_simplify():
    fbx_path = r"C:\Projects\sih\Building-3d_mesh.fbx"
    obj_path = r"C:\Projects\sih\Building-3d_mesh_simplified.obj"
    
    print(f"Loading {fbx_path}...")
    mesh = o3d.io.read_triangle_mesh(fbx_path)
    
    # 1. Simplify the mesh because 1,000,000 faces will take forever in python raycasting
    target_faces = 50000
    print(f"Original mesh: {len(mesh.triangles)} faces.")
    print(f"Simplifying to {target_faces} faces...")
    
    mesh_smp = mesh.simplify_quadric_decimation(target_faces)
    print(f"Simplified to {len(mesh_smp.triangles)} faces.")
    
    # 2. Save as OBJ
    print(f"Saving to {obj_path}...")
    o3d.io.write_triangle_mesh(obj_path, mesh_smp)
    print("Done!")

if __name__ == "__main__":
    convert_and_simplify()
