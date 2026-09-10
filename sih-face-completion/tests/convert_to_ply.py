import open3d as o3d
from pathlib import Path

def convert_to_ply():
    obj_path = r"tests_output\pix4d_test\scene_completed_colored.obj"
    ply_path = r"tests_output\pix4d_test\scene_completed_colored.ply"
    
    mesh = o3d.io.read_triangle_mesh(obj_path)
    print("Exporting as PLY so Windows 3D viewer can see the vertex colors...")
    o3d.io.write_triangle_mesh(ply_path, mesh, write_vertex_normals=True, write_vertex_colors=True)
    print("Done")

if __name__ == "__main__":
    convert_to_ply()
