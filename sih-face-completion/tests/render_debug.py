import trimesh
import numpy as np

def render_mesh():
    mesh = trimesh.load(r"tests_output\pix4d_test\scene_completed_colored.obj")
    
    # Setup scene
    scene = trimesh.Scene(mesh)
    
    # Set a nice camera angle
    # We want to look down at the building
    scene.camera_transform = scene.camera.look_at(
        points=[mesh.centroid],
        center=mesh.centroid,
        distance=max(mesh.extents) * 1.2
    )
    
    # Render
    data = scene.save_image(resolution=(1024, 768))
    with open("tests_output/pix4d_test/debug_render.png", "wb") as f:
        f.write(data)
    print("Rendered to tests_output/pix4d_test/debug_render.png")

if __name__ == "__main__":
    render_mesh()
