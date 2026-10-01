import os
import sys
import numpy as np
import pyproj
import open3d as o3d
from typing import Dict, List, Any, Optional, Tuple
from reconstruction_engine.logger import PipelineLogger, PipelineTracker

class GeoreferencingEngine:
    """
    Step 10: 7-Parameter Helmert Georeferencing & Accuracy Validation.
    Aligns reconstructed model to metric UTM coordinates (WGS84) via Umeyama SVD.
    Validates flight trajectory conditioning before trusting fit and computes RMS residual.
    """
    def __init__(self, config: Dict[str, Any], logger: PipelineLogger, tracker: PipelineTracker):
        self.config = config.get("step10_georeferencing", config)
        self.logger = logger
        self.tracker = tracker

    def georeference_scene(
        self,
        camera_poses: Dict[str, Dict],
        frames: List[Dict[str, Any]],
        gcps: Optional[List[Dict[str, Any]]],
        pcd: o3d.geometry.PointCloud,
        mesh: o3d.geometry.TriangleMesh,
        workspace_dir: str
    ) -> Dict[str, Any]:
        """
        Executes georeferencing transform.
        Returns:
            Dict containing:
                - 'sim_transform': 4x4 similarity transformation matrix
                - 'rms_residual_m': float (in meters)
                - 'crs_epsg': str (e.g. 'EPSG:32643')
                - 'conditioning_status': str
                - 'conditioning_warning': Optional[str]
        """
        self.logger.stage_header(10, "Georeferencing & Absolute Metric Alignment")

        # 1. Gather Camera Centers in Model Space and Matching Real-World GPS Coordinates
        model_pts = []
        gps_wgs84 = []
        weights = []

        frame_dict = {f["filename"]: f for f in frames}

        for cam_name, pose in camera_poses.items():
            if cam_name in frame_dict and frame_dict[cam_name].get("gps"):
                g = frame_dict[cam_name]["gps"]
                model_pts.append(pose["center"])
                gps_wgs84.append([g["lon"], g["lat"], g["alt"]])
                weights.append(1.0)

        # Ingest GCPs if available (higher weight)
        if gcps and len(gcps) > 0:
            for gcp in gcps:
                # If image match exists
                if gcp.get("image") in camera_poses:
                    # Treat GCP position with high weight
                    gps_wgs84.append([gcp["lon"], gcp["lat"], gcp["alt"]])
                    model_pts.append(camera_poses[gcp["image"]]["center"])
                    weights.append(10.0)

        if len(model_pts) < 3:
            msg = "Fewer than 3 matched GPS camera priors; cannot solve 7-parameter Helmert similarity."
            self.tracker.mark_degraded("step10_georeferencing", msg, self.logger)
            self.tracker.metrics["crs_epsg"] = "LOCAL_METRIC"
            self.tracker.metrics["georef_rms_residual_m"] = 0.0
            return {
                "sim_transform": np.eye(4),
                "rms_residual_m": 0.0,
                "crs_epsg": "LOCAL_METRIC",
                "conditioning_status": "LOCAL_UNALIGNED",
                "conditioning_warning": msg
            }

        model_pts = np.array(model_pts)
        gps_wgs84 = np.array(gps_wgs84)
        weights = np.array(weights)

        # 2. Determine Optimal UTM Zone & EPSG Code
        avg_lon = float(np.mean(gps_wgs84[:, 0]))
        avg_lat = float(np.mean(gps_wgs84[:, 1]))
        utm_zone = int((avg_lon + 180.0) / 6.0) + 1
        is_northern = avg_lat >= 0.0
        epsg_code = 32600 + utm_zone if is_northern else 32700 + utm_zone
        crs_str = f"EPSG:{epsg_code}"
        self.logger.info(f"Target Coordinate Reference System: {crs_str} (UTM Zone {utm_zone}{'N' if is_northern else 'S'}, WGS 84).")

        # Project WGS84 (lon, lat, alt) -> UTM (easting, northing, alt)
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
        eastings, northings = transformer.transform(gps_wgs84[:, 0], gps_wgs84[:, 1])
        utm_pts = np.column_stack([eastings, northings, gps_wgs84[:, 2]])

        # 3. Check Trajectory Conditioning Before Trusting Fit
        cond_info = self.tracker.metrics.get("geometry_conditioning", {})
        cond_status = cond_info.get("status", "UNKNOWN")
        cond_warning = cond_info.get("warning")

        if cond_status == "POORLY_CONDITIONED_NEAR_LINEAR" and (not gcps or len(gcps) < 3):
            self.logger.warning(
                "CAUTION: Flight path is near-linear without off-axis GCPs! "
                "7-parameter Helmert rotation around flight axis is underconstrained. "
                "RMS residual indicates 2D trajectory alignment but rotational uncertainty along flight axis remains."
            )

        # 4. Compute Optimal 7-Parameter Similarity Transform (Umeyama Algorithm)
        sim_transform, s, R, t = self._umeyama_alignment(model_pts, utm_pts, weights)

        # 5. Compute Real-World RMS Residual (Meters)
        transformed_model_pts = (s * (R @ model_pts.T) + t[:, None]).T
        residuals = np.linalg.norm(transformed_model_pts - utm_pts, axis=1)
        rms_residual = float(np.sqrt(np.mean(residuals**2)))
        self.tracker.metrics["georef_rms_residual_m"] = round(rms_residual, 3)
        self.tracker.metrics["crs_epsg"] = crs_str

        max_acc = float(self.config.get("max_acceptable_rms_m", 5.0))
        if rms_residual > max_acc:
            self.logger.warning(f"Georeferencing RMS residual ({rms_residual:.2f}m) exceeds threshold ({max_acc:.2f}m)!")
        else:
            self.logger.success(f"Helmert alignment successful. Real-world RMS residual: {rms_residual:.3f} meters across {len(model_pts)} points.")

        # 6. Transform Point Cloud and Mesh to Metric UTM Space
        # (To prevent single-precision float jitter in 3D viewers, models are exported with translation origin metadata)
        pcd.transform(sim_transform)
        mesh.transform(sim_transform)

        return {
            "sim_transform": sim_transform,
            "scale": s,
            "rotation": R,
            "translation": t,
            "rms_residual_m": rms_residual,
            "crs_epsg": crs_str,
            "utm_zone": utm_zone,
            "conditioning_status": cond_status,
            "conditioning_warning": cond_warning
        }

    def _umeyama_alignment(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Calculates optimal similarity transformation (s, R, t) mapping X to Y:
        min sum w_i || s * R * X_i + t - Y_i ||^2
        """
        w = weights / np.sum(weights)
        mu_x = np.sum(X * w[:, None], axis=0)
        mu_y = np.sum(Y * w[:, None], axis=0)

        X_c = X - mu_x
        Y_c = Y - mu_y

        var_x = np.sum(w[:, None] * (X_c ** 2))

        # Weighted covariance
        Sigma = (Y_c.T * w) @ X_c

        U, D, Vt = np.linalg.svd(Sigma)
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1

        R = U @ S @ Vt
        s = float((np.trace(np.diag(D) @ S)) / (var_x + 1e-12))
        t = mu_y - s * (R @ mu_x)

        # 4x4 matrix
        T_mat = np.eye(4)
        T_mat[:3, :3] = s * R
        T_mat[:3, 3] = t

        return T_mat, s, R, t
