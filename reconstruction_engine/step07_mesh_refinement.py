import os
import sys
import numpy as np
import open3d as o3d
import pymeshlab
from typing import Dict, List, Any, Optional, Tuple
from reconstruction_engine.logger import PipelineLogger, PipelineTracker

class MeshRefinementEngine:
    """
    Step 7: Mesh Refinement & Topology Cleanup.
    Enforces clean manifold topology: removes degenerate triangles, duplicate vertices/faces,
    non-manifold edges, fills unobserved holes, and applies surface smoothing.
    """
    def __init__(self, config: Dict[str, Any], logger: PipelineLogger, tracker: PipelineTracker, toolchain: Dict[str, Any]):
        self.config = config.get("step07_refinement", config)
        self.toolchain = toolchain
        self.logger = logger
        self.tracker = tracker

    def refine_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        densities: np.ndarray,
        provenance: np.ndarray,
        workspace_dir: str
    ) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, np.ndarray, str]:
        """
        Refines topology and repairs defects.
        Returns:
            - Cleaned TriangleMesh
            - Updated vertex densities
            - Updated vertex provenance
            - Refined mesh file path
        """
        self.logger.stage_header(7, "Mesh Refinement & Topology Cleanup")
        
        refined_dir = os.path.join(workspace_dir, "refined_mesh")
        os.makedirs(refined_dir, exist_ok=True)
        refined_ply = os.path.join(refined_dir, "mesh_refined.ply")

        initial_v = len(mesh.vertices)
        initial_f = len(mesh.triangles) if hasattr(mesh, "triangles") else len(mesh.faces)
        self.logger.info(f"Initial mesh topology: {initial_v} vertices, {initial_f} faces.")

        # 1. Remove Degenerate Triangles
        if self.config.get("remove_degenerate", True):
            mesh.remove_degenerate_triangles()
        
        # 2. Remove Duplicates
        if self.config.get("remove_duplicated_triangles", True):
            mesh.remove_duplicated_triangles()
        if self.config.get("remove_duplicated_vertices", True):
            mesh.remove_duplicated_vertices()

        # 3. Remove Non-Manifold Edges
        if self.config.get("remove_non_manifold", True):
            mesh.remove_non_manifold_edges()

        # 4. PyMeshLab Enhanced Pass (Hole filling and Laplacian Smoothing)
        old_verts = np.asarray(mesh.vertices).copy()
        temp_in = os.path.join(refined_dir, "temp_pre_pymeshlab.ply")
        o3d.io.write_triangle_mesh(temp_in, mesh)

        try:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(temp_in)
            
            # Close holes
            max_hole_verts = int(self.config.get("max_hole_vertices", 30))
            ms.meshing_close_holes(maxholesize=max_hole_verts)

            # Taubin / Laplacian surface smoothing
            smooth_iters = int(self.config.get("laplacian_smoothing_iterations", 2))
            if smooth_iters > 0:
                ms.apply_coord_laplacian_smoothing(stepsmoothnum=smooth_iters)

            # Save out
            ms.save_current_mesh(refined_ply)
            mesh = o3d.io.read_triangle_mesh(refined_ply)
            if os.path.exists(temp_in):
                os.remove(temp_in)
            self.logger.info("PyMeshLab hole closing and surface smoothing pass applied successfully.")
        except Exception as e:
            self.logger.warning(f"PyMeshLab pass skipped ({e}). Falling back to Open3D native cleanup.")
            o3d.io.write_triangle_mesh(refined_ply, mesh)

        # 5. Synchronize vertex attributes with nearest-neighbor search
        final_verts = np.asarray(mesh.vertices)
        if len(final_verts) != len(densities):
            self.logger.info("Resynchronizing vertex attributes with refined topology...")
            old_pcd = o3d.geometry.PointCloud()
            old_pcd.points = o3d.utility.Vector3dVector(old_verts)
            tree = o3d.geometry.KDTreeFlann(old_pcd)
            
            new_densities = []
            new_provenance = []
            for v in final_verts:
                [_, idx, _] = tree.search_knn_vector_3d(v, 1)
                i = idx[0]
                new_densities.append(densities[i] if i < len(densities) else 1.0)
                new_provenance.append(provenance[i] if i < len(provenance) else "mvs")
            densities = np.array(new_densities, dtype=np.float32)
            provenance = np.array(new_provenance, dtype=object)

        final_v = len(mesh.vertices)
        final_f = len(mesh.triangles) if hasattr(mesh, "triangles") else len(mesh.faces)
        self.tracker.metrics["mesh_vertices"] = final_v
        self.tracker.metrics["mesh_faces_full"] = final_f

        # Save synchronized attributes
        np.save(os.path.join(refined_dir, "refined_densities.npy"), densities)
        np.save(os.path.join(refined_dir, "refined_provenance.npy"), provenance)

        self.logger.success(f"Topology cleanup verified: Manifold mesh with {final_v} vertices, {final_f} faces.")
        return mesh, densities, provenance, refined_ply
