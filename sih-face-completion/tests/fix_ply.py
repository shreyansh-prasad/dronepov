import open3d as o3d
import trimesh
import numpy as np

def fix_ply():
    print("Loading OBJ with Open3D...")
    mesh = o3d.io.read_triangle_mesh(r"tests_output\pix4d_test\scene_completed_colored.obj")
    
    print("Converting colors to 8-bit RGBA...")
    colors = np.asarray(mesh.vertex_colors)
    colors_u8 = (colors * 255.0).clip(0, 255).astype(np.uint8)
    colors_rgba = np.hstack([colors_u8, np.full((len(colors_u8), 1), 255, dtype=np.uint8)])
    
    print("Building standard Trimesh object...")
    tm = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                         faces=np.asarray(mesh.triangles),
                         vertex_colors=colors_rgba,
                         vertex_normals=np.asarray(mesh.vertex_normals))
    
    print("Exporting standard PLY...")
    tm.export(r"tests_output\pix4d_test\scene_completed_colored.ply")
    print("Fixed PLY colors!")
    
if __name__ == "__main__":
    fix_ply()
