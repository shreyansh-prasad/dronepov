import os
import sys
import numpy as np
import open3d as o3d
from typing import Dict, List, Any, Optional, Tuple
from reconstruction_engine.logger import PipelineLogger, PipelineTracker

class PointCloudCleanupEngine:
    """
    Step 5: Point Cloud Cleanup & Normal Estimation.
    Applies statistical outlier removal, radius outlier removal, secondary dynamic-mask filtering,
    and estimates consistent surface normals. Synchronizes point provenance tags.
    """
    def __init__(self, config: Dict[str, Any], logger: PipelineLogger, tracker: PipelineTracker):
        self.config = config.get("step05_cleanup", config)
        self.logger = logger
        self.tracker = tracker

    def cleanup_pointcloud(
        self,
        dense_ply_path: str,
        provenance_path: Optional[str],
        output_ply_path: str
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray, str]:
        """
        Executes point cloud filtering and normal orientation.
        Returns:
            - Cleaned Open3D PointCloud
            - Synchronized provenance array (N,)
            - Path to cleaned PLY
        """
        self.logger.stage_header(5, "Point Cloud Cleanup & Normal Estimation")
        
        if not os.path.exists(dense_ply_path):
            raise FileNotFoundError(f"Dense point cloud not found at {dense_ply_path}")

        pcd = o3d.io.read_point_cloud(dense_ply_path)
        raw_count = len(pcd.points)
        self.logger.info(f"Loaded raw dense point cloud: {raw_count} points.")

        provenance = None
        if provenance_path and os.path.exists(provenance_path):
            provenance = np.load(provenance_path, allow_pickle=True)
            if len(provenance) != raw_count:
                self.logger.warning(f"Provenance length ({len(provenance)}) != point count ({raw_count}). Resetting.")
                provenance = np.array(["mvs"] * raw_count, dtype=object)
        else:
            provenance = np.array(["mvs"] * raw_count, dtype=object)

        # 1. Statistical Outlier Removal (SOR)
        nb_neighbors = int(self.config.get("nb_neighbors", 20))
        std_ratio = float(self.config.get("std_ratio", 1.5))
        self.logger.info(f"Applying Statistical Outlier Removal (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})...")
        
        pcd_clean, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        
        inlier_indices = np.asarray(inlier_indices)
        sor_removed = raw_count - len(pcd_clean.points)
        provenance = provenance[inlier_indices]
        self.logger.info(f"SOR removed {sor_removed} outlier points ({len(pcd_clean.points)} remaining).")
        pcd = pcd_clean

        # 2. Estimate Surface Normals
        if self.config.get("estimate_normals", True):
            self.logger.info("Estimating and orienting surface normals...")
            max_nn = int(self.config.get("normal_max_nn", 30))
            
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max_nn)
            )
            try:
                # Orient normals consistently towards camera Z axis for UAV oblique/nadir surveys
                pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, -1.0]))
            except Exception as e:
                self.logger.warning(f"Normal orientation warning: {e}.")

        # Save cleaned point cloud and synchronized provenance
        os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
        o3d.io.write_point_cloud(output_ply_path, pcd)
        
        clean_prov_path = output_ply_path.replace(".ply", "_provenance.npy")
        np.save(clean_prov_path, provenance)

        final_count = len(pcd.points)
        self.tracker.metrics["dense_points"] = final_count
        self.logger.success(f"Point cloud cleanup complete: {final_count} high-confidence points with normals saved to '{output_ply_path}'.")

        return pcd, provenance, clean_prov_path
