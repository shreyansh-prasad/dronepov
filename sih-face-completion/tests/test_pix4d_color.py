import open3d as o3d
import numpy as np
from pathlib import Path

def run_poisson_with_colors():
    fbx_path = r"C:\Projects\sih\Building-3d_mesh.fbx"
    out_dir = Path("tests_output/pix4d_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / "scene_completed_colored.obj"
    
    print(f"Loading {fbx_path}...")
    mesh = o3d.io.read_triangle_mesh(fbx_path)
    
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    if mesh.has_textures() and mesh.has_triangle_uvs():
        print("Baking 8K texture into vertex colors...")
        tex = np.asarray(mesh.textures[0])
        h, w, c = tex.shape
        
        uvs = np.asarray(mesh.triangle_uvs) 
        triangles = np.asarray(mesh.triangles).flatten()
        
        u = uvs[:, 0]
        v = uvs[:, 1]
        
        px = np.clip((u * (w - 1)).astype(np.int32), 0, w - 1)
        py = np.clip(((1.0 - v) * (h - 1)).astype(np.int32), 0, h - 1)
        sampled_colors = tex[py, px, :3] / 255.0
        
        vertex_colors = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
        # Direct assignment is thousands of times faster than np.add.at
        vertex_colors[triangles] = sampled_colors
        
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    else:
        print("Mesh does not have textures!")

    print("Running Poisson Surface Reconstruction...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals
    if mesh.has_vertex_colors():
        pcd.colors = mesh.vertex_colors
    
    mesh_out, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh_out = mesh_out.crop(bbox)

    o3d.io.write_triangle_mesh(str(obj_path), mesh_out, write_vertex_normals=True, write_vertex_colors=True)
    print("Done! You can now view the colored mesh.")

if __name__ == "__main__":
    run_poisson_with_colors()
