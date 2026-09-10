import open3d as o3d
import numpy as np

def test_bake():
    fbx_path = r"C:\Projects\sih\Building-3d_mesh.fbx"
    out_path = r"tests_output\pix4d_test\baked_original.ply"
    
    mesh = o3d.io.read_triangle_mesh(fbx_path)
    
    tex = np.asarray(mesh.textures[0])
    h, w, c = tex.shape
    
    uvs = np.asarray(mesh.triangle_uvs) 
    triangles = np.asarray(mesh.triangles).flatten()
    
    u = uvs[:, 0]
    
    # Try V without flipping
    v1 = uvs[:, 1]
    px = np.clip((u * (w - 1)).astype(np.int32), 0, w - 1)
    py1 = np.clip((v1 * (h - 1)).astype(np.int32), 0, h - 1)
    
    # Try V with flipping (standard OpenGL)
    v2 = 1.0 - uvs[:, 1]
    py2 = np.clip((v2 * (h - 1)).astype(np.int32), 0, h - 1)
    
    # We will save the mesh with py2 (flipped) which is what I did before
    # Wait, let's try py1 (unflipped) since py2 resulted in a gray mess for the user!
    sampled_colors = tex[py1, px, :3] / 255.0
    
    vertex_colors = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
    vertex_colors[triangles] = sampled_colors
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    o3d.io.write_triangle_mesh(out_path, mesh, write_vertex_normals=True, write_vertex_colors=True)
    print(f"Saved unflipped V to {out_path}")

if __name__ == "__main__":
    test_bake()
