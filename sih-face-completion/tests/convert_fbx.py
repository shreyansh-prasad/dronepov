import open3d as o3d
import numpy as np

def convert_fbx_to_colored_obj():
    fbx_path = r"C:\Projects\sih\Building-3d_mesh.fbx"
    obj_path = r"C:\Projects\sih\Building-3d_mesh_colored.obj"
    
    print(f"Loading {fbx_path}...")
    mesh = o3d.io.read_triangle_mesh(fbx_path)
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
    
    if mesh.has_textures() and mesh.has_triangle_uvs():
        print("Baking texture to vertex colors...")
        tex = np.asarray(mesh.textures[0])
        h, w, c = tex.shape
        
        uvs = np.asarray(mesh.triangle_uvs) # (N*3, 2)
        triangles = np.asarray(mesh.triangles).flatten() # (N*3,)
        
        u = uvs[:, 0]
        v = uvs[:, 1]
        
        px = np.clip((u * (w - 1)).astype(np.int32), 0, w - 1)
        py = np.clip(((1.0 - v) * (h - 1)).astype(np.int32), 0, h - 1)
        
        sampled_colors = tex[py, px, :3] / 255.0
        
        vertex_colors = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
        counts = np.zeros(len(mesh.vertices), dtype=np.int32)
        
        np.add.at(vertex_colors, triangles, sampled_colors)
        np.add.at(counts, triangles, 1)
        
        valid = counts > 0
        vertex_colors[valid] /= counts[valid][:, None]
        
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        print("Successfully baked texture to vertex colors.")
    
    print(f"Saving to {obj_path}...")
    o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_normals=True, write_vertex_colors=True)
    print("Done!")

if __name__ == "__main__":
    convert_fbx_to_colored_obj()
