import os
import sys
import shutil
import subprocess
import numpy as np
import open3d as o3d
from typing import Dict, List, Any, Optional, Tuple
from reconstruction_engine.logger import PipelineLogger, PipelineTracker

class SurfaceMeshingEngine:
    """
    Step 6: Surface Reconstruction (Meshing).
    Primary: OpenMVS ReconstructMesh (Delaunay tetrahedralization).
    Fallback: Screened Poisson Surface Reconstruction (Open3D) with density preservation.
    """
    def __init__(self, config: Dict[str, Any], logger: PipelineLogger, tracker: PipelineTracker, toolchain: Dict[str, Any]):
        self.config = config.get("step06_meshing", config)
        self.toolchain = toolchain
        self.logger = logger
        self.tracker = tracker
        self.openmvs_bin = self._resolve_openmvs()

    def _resolve_openmvs(self) -> Optional[str]:
        openmvs_dir = self.toolchain.get("openmvs_bin_dir", "")
        if openmvs_dir:
            exe = os.path.join(openmvs_dir, "ReconstructMesh.exe" if sys.platform == "win32" else "ReconstructMesh")
            if os.path.exists(exe):
                return exe
        which = shutil.which("ReconstructMesh")
        return which

    def reconstruct_surface(
        self,
        pcd: o3d.geometry.PointCloud,
        provenance: np.ndarray,
        workspace_dir: str
    ) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, np.ndarray, str]:
        """
        Reconstructs 3D triangle mesh from point cloud.
        Returns:
            - TriangleMesh
            - Vertex densities (N_verts,)
            - Vertex provenance (N_verts,)
            - Output mesh file path
        """
        self.logger.stage_header(6, "Surface Reconstruction (3D Meshing)")
        
        mesh_dir = os.path.join(workspace_dir, "meshing")
        os.makedirs(mesh_dir, exist_ok=True)
        raw_mesh_path = os.path.join(mesh_dir, "mesh_raw.ply")

        # Check OpenMVS primary vs Poisson fallback
        method = self.config.get("method", "auto")
        if method in ("auto", "openmvs") and self.openmvs_bin:
            self.logger.info(f"Executing OpenMVS ReconstructMesh at {self.openmvs_bin}...")
            try:
                # Check for existing valid OpenMVS mesh or run ReconstructMesh
                mvs_scene = os.path.abspath(os.path.join(workspace_dir, "dense_mvs", "scene_dense.mvs"))
                if not os.path.exists(mvs_scene):
                    mvs_scene = os.path.abspath(os.path.join(workspace_dir, "dense_mvs", "scene.mvs"))
                out_mvs = os.path.abspath(os.path.join(mesh_dir, "scene_mesh.mvs"))
                out_ply = os.path.abspath(os.path.join(mesh_dir, "scene_mesh.ply"))
                dense_mvs_mesh_ply = os.path.abspath(os.path.join(workspace_dir, "dense_mvs", "scene_mesh.ply"))

                if os.path.exists(dense_mvs_mesh_ply) and os.path.getsize(dense_mvs_mesh_ply) > 1000:
                    self.logger.info(f"Using existing OpenMVS reconstructed mesh from '{dense_mvs_mesh_ply}'.")
                    shutil.copy2(dense_mvs_mesh_ply, raw_mesh_path)
                elif os.path.exists(mvs_scene):
                    cmd = [self.openmvs_bin, mvs_scene, "-o", out_mvs, "-v", "3"]
                    res = subprocess.run(cmd, cwd=mesh_dir, capture_output=True, text=True)
                    if res.returncode != 0:
                        raise RuntimeError(res.stderr or res.stdout)
                    if os.path.exists(out_ply):
                        shutil.copy2(out_ply, raw_mesh_path)
                    elif os.path.exists(dense_mvs_mesh_ply):
                        shutil.copy2(dense_mvs_mesh_ply, raw_mesh_path)
                    else:
                        raise FileNotFoundError(f"OpenMVS mesh output not found at {out_ply}")
                else:
                    raise FileNotFoundError(f"Neither scene_dense.mvs nor scene_mesh.ply found in {workspace_dir}/dense_mvs")

                mesh = o3d.io.read_triangle_mesh(raw_mesh_path)
                if len(mesh.vertices) == 0:
                    raise ValueError("Reconstructed mesh has 0 vertices.")
                densities = np.ones(len(mesh.vertices), dtype=np.float32)
                self.logger.success(f"Loaded OpenMVS Delaunay graph-cut mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles.")
            except Exception as e:
                self.logger.warning(f"OpenMVS meshing failed or scene.mvs unavailable: {e}. Falling back to Open3D Poisson.")
                self.tracker.mark_degraded("step06_meshing", f"OpenMVS meshing failed ({e}); running Open3D Poisson meshing fallback.", self.logger)
                mesh, densities = self._poisson_meshing(pcd)
        else:
            reason = "OpenMVS ReconstructMesh binary not found; running Open3D Poisson meshing fallback."
            self.tracker.mark_degraded("step06_meshing", reason, self.logger)
            mesh, densities = self._poisson_meshing(pcd)

        # Transfer point provenance to mesh vertices via KDTree lookup
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        vertex_provenance = []
        mesh_verts = np.asarray(mesh.vertices)
        
        for v in mesh_verts:
            [_, idx, _] = pcd_tree.search_knn_vector_3d(v, 1)
            pt_idx = idx[0]
            vertex_provenance.append(provenance[pt_idx] if pt_idx < len(provenance) else "mvs")
            
        vertex_prov_arr = np.array(vertex_provenance, dtype=object)

        # Write raw mesh
        o3d.io.write_triangle_mesh(raw_mesh_path, mesh)
        
        # Save vertex attributes alongside
        np.save(os.path.join(mesh_dir, "vertex_densities.npy"), densities)
        np.save(os.path.join(mesh_dir, "vertex_provenance.npy"), vertex_prov_arr)

        num_faces = len(mesh.triangles) if hasattr(mesh, "triangles") else len(mesh.faces)
        self.logger.success(f"Surface reconstruction complete: {len(mesh.vertices)} vertices, {num_faces} faces.")
        return mesh, densities, vertex_prov_arr, raw_mesh_path

    def _poisson_meshing(self, pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        """Screened Poisson surface reconstruction with density trimming."""
        depth = int(self.config.get("poisson_depth", 9))
        scale = float(self.config.get("poisson_scale", 1.1))
        self.logger.info(f"Running Screened Poisson Surface Reconstruction (depth={depth}, scale={scale})...")

        # Ensure normals exist
        if not pcd.has_normals():
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(15)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            scale=scale,
            linear_fit=True
        )

        densities = np.asarray(densities)
        
        # Trim low-density boundary artifacts (extrapolated ballooning outside observation volume)
        density_thresh = np.quantile(densities, 0.05)
        vertices_to_remove = densities < density_thresh
        mesh.remove_vertices_by_mask(vertices_to_remove)
        densities = densities[~vertices_to_remove]

        self.logger.info(f"Poisson meshing trimmed {np.sum(vertices_to_remove)} low-density boundary vertices.")
        return mesh, densities
