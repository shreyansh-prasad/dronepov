import os
import sys
import shutil
import subprocess
import numpy as np
from PIL import Image
import open3d as o3d
import trimesh
import xatlas
from typing import Dict, List, Any, Optional, Tuple
from reconstruction_engine.logger import PipelineLogger, PipelineTracker

class TextureMappingEngine:
    """
    Step 8: UV Texturing from Source Video Frames.
    Primary: OpenMVS TextureMesh (photogrammetric multi-view texturing).
    Fallback: Best-view-per-triangle camera projection with xatlas UV unwrapping and atlas packing.
    Generates mesh_full.obj (+ .mtl + .png) and self-contained mesh_full.glb.
    """
    def __init__(self, config: Dict[str, Any], logger: PipelineLogger, tracker: PipelineTracker, toolchain: Dict[str, Any]):
        self.config = config.get("step08_texturing", config)
        self.toolchain = toolchain
        self.logger = logger
        self.tracker = tracker
        self.texture_mesh_bin = self._resolve_texture_mesh()

    def _resolve_texture_mesh(self) -> Optional[str]:
        openmvs_dir = self.toolchain.get("openmvs_bin_dir", "")
        if openmvs_dir:
            exe = os.path.join(openmvs_dir, "TextureMesh.exe" if sys.platform == "win32" else "TextureMesh")
            if os.path.exists(exe):
                return exe
        return shutil.which("TextureMesh")

    def texture_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        frames: List[Dict[str, Any]],
        camera_poses: Dict[str, Dict],
        workspace_dir: str,
        output_prefix: str
    ) -> Dict[str, str]:
        """
        Applies photographic UV texturing.
        Returns:
            Dict containing:
                - 'obj_path': path to mesh_full.obj
                - 'mtl_path': path to mesh_full.mtl
                - 'texture_path': path to mesh_full.png
                - 'glb_path': path to mesh_full.glb
        """
        self.logger.stage_header(8, "Photographic UV Texture Mapping")
        
        texture_dir = os.path.join(workspace_dir, "textured_mesh")
        os.makedirs(texture_dir, exist_ok=True)
        
        obj_out = os.path.join(texture_dir, f"{output_prefix}.obj")
        mtl_out = os.path.join(texture_dir, f"{output_prefix}.mtl")
        png_out = os.path.join(texture_dir, f"{output_prefix}.png")
        glb_out = os.path.join(texture_dir, f"{output_prefix}.glb")

        method = self.config.get("method", "auto")
        if method in ("auto", "openmvs") and self.texture_mesh_bin:
            self.logger.info(f"Executing OpenMVS TextureMesh at {self.texture_mesh_bin}...")
            try:
                dense_dir = os.path.join(workspace_dir, "dense_mvs")
                scene_dense_mvs = os.path.join(dense_dir, "scene_dense.mvs")
                if not os.path.exists(scene_dense_mvs):
                    scene_dense_mvs = os.path.join(dense_dir, "scene.mvs")

                mesh_ply = os.path.join(workspace_dir, "refined_mesh", "mesh_refined.ply")
                if not os.path.exists(mesh_ply):
                    mesh_ply = os.path.join(workspace_dir, "meshing", "mesh_raw.ply")
                if not os.path.exists(mesh_ply):
                    mesh_ply = os.path.join(dense_dir, "scene_mesh.ply")

                existing_dense_glb = os.path.join(dense_dir, "scene_textured.glb")
                existing_dense_obj = os.path.join(dense_dir, "scene_textured.obj")

                if os.path.exists(existing_dense_glb) and os.path.exists(existing_dense_obj) and os.path.getsize(existing_dense_glb) > 1000:
                    self.logger.info("Using existing OpenMVS textured GLB and OBJ models.")
                    shutil.copy2(existing_dense_glb, glb_out)
                    shutil.copy2(existing_dense_obj, obj_out)
                    for ext in [".mtl", "_material_00_map_Kd.jpg", "_0.png"]:
                        src = os.path.join(dense_dir, f"scene_textured{ext}")
                        if os.path.exists(src):
                            shutil.copy2(src, texture_dir)
                    dense_mtl = os.path.join(dense_dir, "scene_textured.mtl")
                    if os.path.exists(dense_mtl):
                        shutil.copy2(dense_mtl, mtl_out)
                elif os.path.exists(scene_dense_mvs) and os.path.exists(mesh_ply):
                    # Run TextureMesh for GLB
                    cmd_glb = [self.texture_mesh_bin, os.path.abspath(scene_dense_mvs), "-m", os.path.abspath(mesh_ply), "--export-type", "glb", "-o", os.path.abspath(glb_out), "-v", "3"]
                    subprocess.run(cmd_glb, cwd=texture_dir, check=True, capture_output=True)

                    # Run TextureMesh for OBJ
                    cmd_obj = [self.texture_mesh_bin, os.path.abspath(scene_dense_mvs), "-m", os.path.abspath(mesh_ply), "--export-type", "obj", "-o", os.path.abspath(obj_out), "-v", "3"]
                    subprocess.run(cmd_obj, cwd=texture_dir, check=True, capture_output=True)
                else:
                    raise FileNotFoundError("Could not find scene_dense.mvs and mesh_refined.ply for texturing.")

                # Locate generated texture image in texture_dir
                tex_candidates = [f for f in os.listdir(texture_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg')) and not f.startswith("temp")]
                if tex_candidates:
                    chosen_tex = os.path.join(texture_dir, tex_candidates[0])
                    if not os.path.exists(png_out) or os.path.getsize(png_out) == 0:
                        shutil.copy2(chosen_tex, png_out)
                self.logger.success("OpenMVS photographic UV texturing completed successfully.")
            except Exception as e:
                self.logger.warning(f"OpenMVS TextureMesh failed ({e}). Falling back to xatlas projection texturing.")
                self.tracker.mark_degraded("step08_texturing", f"OpenMVS TextureMesh failed ({e}); running xatlas UV camera projection fallback.", self.logger)
                self._xatlas_projection_texturing(mesh, frames, camera_poses, obj_out, mtl_out, png_out, glb_out)
        else:
            reason = "OpenMVS TextureMesh binary not found; running xatlas UV camera projection fallback."
            self.tracker.mark_degraded("step08_texturing", reason, self.logger)
            self._xatlas_projection_texturing(mesh, frames, camera_poses, obj_out, mtl_out, png_out, glb_out)

        # -------------------------------------------------------------------
        # Ensure the final GLB is self‑contained by re‑exporting via Trimesh.
        # This loads the generated OBJ (which references the PNG) and writes a new
        # GLB with the texture embedded in the binary buffers.
        try:
            self.logger.info("Re‑exporting GLB with embedded texture using Trimesh...")
            # Load the OBJ mesh (including UVs)
            mesh_obj = trimesh.load(str(obj_out), force='mesh')
            # Extract UV coordinates from the mesh visual
            uv_coords = getattr(mesh_obj.visual, 'uv', None)
            if uv_coords is None:
                raise RuntimeError('UV coordinates not found in loaded OBJ.')
            # Load the texture image we saved earlier
            texture_image = Image.open(png_out)
            # Create a TextureVisuals object with the UVs and image
            visual = trimesh.visual.TextureVisuals(uv=uv_coords, image=texture_image)
            # Construct a new Trimesh with embedded texture
            tm_mesh = trimesh.Trimesh(vertices=mesh_obj.vertices,
                                      faces=mesh_obj.faces,
                                      visual=visual,
                                      process=False)
            # Export the GLB with the texture embedded
            glb_bytes = tm_mesh.export(file_type='glb')
            with open(glb_out, 'wb') as f_out:
                f_out.write(glb_bytes)
            self.logger.success("Self‑contained GLB written.")
        except Exception as e:
            self.logger.warning(f"Failed to embed texture in GLB: {e}. Keeping original GLB.")

        return {
            "obj_path": obj_out,
            "mtl_path": mtl_out,
            "texture_path": png_out,
            "glb_path": glb_out
        }

    def _xatlas_projection_texturing(
        self,
        mesh: o3d.geometry.TriangleMesh,
        frames: List[Dict[str, Any]],
        camera_poses: Dict[str, Dict],
        obj_path: str,
        mtl_path: str,
        png_path: str,
        glb_path: str
    ):
        """
        Fallback UV texturing engine:
        1. Uses xatlas to unwrap mesh parameterization and pack UV charts into an atlas.
        2. Selects the optimal video keyframe for each triangle based on normal-to-camera ray alignment.
        3. Bakes projected imagery into the texture atlas PNG.
        4. Exports OBJ/MTL and binary GLB.
        """
        self.logger.info("Parametrizing mesh and packing UV atlas charts using xatlas...")
        
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles if hasattr(mesh, "triangles") else mesh.faces)

        # 1. Parameterize with xatlas
        vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
        new_vertices = vertices[vmapping]
        new_faces = indices

        atlas_res = int(self.config.get("texture_resolution", 2048))
        atlas_img = np.zeros((atlas_res, atlas_res, 3), dtype=np.uint8)
        
        # Load frame images and intrinsics into memory
        loaded_images = {}
        for f in frames:
            name = f["filename"]
            if name in camera_poses and os.path.exists(f["path"]):
                try:
                    with Image.open(f["path"]) as img:
                        loaded_images[name] = {
                            "rgb": np.array(img.convert("RGB")),
                            "pose": camera_poses[name],
                            "w": img.width,
                            "h": img.height
                        }
                except Exception:
                    continue

        self.logger.info(f"Loaded {len(loaded_images)} camera images for ray projection texturing.")

        # Compute face normals and centroids
        v0 = new_vertices[new_faces[:, 0]]
        v1 = new_vertices[new_faces[:, 1]]
        v2 = new_vertices[new_faces[:, 2]]
        face_centroids = (v0 + v1 + v2) / 3.0
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normals = np.cross(edge1, edge2)
        norm_len = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12
        face_normals = face_normals / norm_len

        # For each face, determine best camera view
        best_cam_per_face = []
        for i in range(len(new_faces)):
            c_f = face_centroids[i]
            n_f = face_normals[i]
            
            best_cam = None
            best_score = -1.0
            
            for cam_name, cam_data in loaded_images.items():
                cam_pos = cam_data["pose"]["center"]
                ray = cam_pos - c_f
                dist = np.linalg.norm(ray) + 1e-6
                ray_dir = ray / dist
                # Angle score: cos theta (facing camera)
                score = np.dot(n_f, ray_dir)
                if score > best_score:
                    best_score = score
                    best_cam = cam_name

            best_cam_per_face.append(best_cam if best_score > 0.1 else (list(loaded_images.keys())[0] if loaded_images else None))

        # 2. Rasterize/bake triangles into UV atlas
        # For performance, sample keyframe colors per UV island
        self.logger.info("Baking photographic camera pixels into UV texture atlas...")
        import cv2

        for i in range(len(new_faces)):
            cam_name = best_cam_per_face[i]
            if not cam_name or cam_name not in loaded_images:
                continue

            cam_data = loaded_images[cam_name]
            img_rgb = cam_data["rgb"]
            R = cam_data["pose"]["R"]
            t = cam_data["pose"]["t"]
            im_w, im_h = cam_data["w"], cam_data["h"]

            # Project centroid to camera image
            c_f = face_centroids[i]
            p_cam = R @ c_f + t
            if p_cam[2] <= 0.1:
                continue
            
            # Simple pinhole projection
            fx = im_w * 0.8
            fy = fx
            cx = im_w / 2.0
            cy = im_h / 2.0
            px = int(np.clip(fx * (p_cam[0] / p_cam[2]) + cx, 0, im_w - 1))
            py = int(np.clip(fy * (p_cam[1] / p_cam[2]) + cy, 0, im_h - 1))
            color = img_rgb[py, px]

            # UV coordinates in atlas pixels
            uv_tri = (uvs[new_faces[i]] * [atlas_res - 1, atlas_res - 1]).astype(np.int32)
            # Fill polygon in atlas
            cv2.fillConvexPoly(atlas_img, uv_tri, [int(color[0]), int(color[1]), int(color[2])])

        # If any empty black regions remain, dilate to avoid black seams
        mask_black = (atlas_img[:, :, 0] == 0) & (atlas_img[:, :, 1] == 0) & (atlas_img[:, :, 2] == 0)
        if np.any(mask_black):
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(atlas_img, kernel)
            atlas_img[mask_black] = dilated[mask_black]

        # Save texture PNG
        Image.fromarray(atlas_img).save(png_path)
        self.logger.info(f"Texture atlas saved to '{png_path}' ({atlas_res}x{atlas_res} px).")

        # 3. Export Wavefront OBJ + MTL
        mtl_filename = os.path.basename(mtl_path)
        png_filename = os.path.basename(png_path)

        with open(mtl_path, "w", encoding="utf-8") as f:
            f.write(f"# UAV Textured Mesh Material\n")
            f.write(f"newmtl material_0\n")
            f.write("Ka 1.0 1.0 1.0\n")
            f.write("Kd 1.0 1.0 1.0\n")
            f.write("Ks 0.0 0.0 0.0\n")
            f.write("d 1.0\n")
            f.write("illum 1\n")
            f.write(f"map_Kd {png_filename}\n")

        with open(obj_path, "w", encoding="utf-8") as f:
            f.write(f"# UAV 3D Reconstruction Model\n")
            f.write(f"mtllib {mtl_filename}\n")
            for x, y, z in new_vertices:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for u, v in uvs:
                f.write(f"vt {u:.6f} {1.0 - v:.6f}\n")
            f.write(f"usemtl material_0\n")
            for f0, f1, f2 in new_faces:
                # 1-indexed for OBJ: v/vt
                f.write(f"f {f0+1}/{f0+1} {f1+1}/{f1+1} {f2+1}/{f2+1}\n")

        # 4. Export self-contained binary GLB using Trimesh
        try:
            visual = trimesh.visual.TextureVisuals(
                uv=uvs,
                image=Image.fromarray(atlas_img)
            )
            tm_mesh = trimesh.Trimesh(
                vertices=new_vertices,
                faces=new_faces,
                visual=visual,
                process=False
            )
            glb_bytes = tm_mesh.export(file_type="glb")
            with open(glb_path, "wb") as f:
                f.write(glb_bytes)
            self.logger.success(f"Full-detail textured mesh exported to '{obj_path}' and '{glb_path}'.")
        except Exception as e:
            self.logger.warning(f"GLB export warning: {e}. Falling back to Open3D export.")
            o3d.io.write_triangle_mesh(glb_path, mesh)
