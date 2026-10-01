import os
import sys
import shutil
import sqlite3
import subprocess
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from reconstruction_engine.logger import PipelineLogger, PipelineTracker

class SparseSfMEngine:
    """
    Step 2: Sparse Structure-from-Motion (SfM) using COLMAP.
    Executes sequential matching with loop detection, incorporates fixed intrinsics if provided,
    applies early GPS-based model alignment priors, and computes geometric conditioning of the flight path.
    """
    def __init__(self, config: Dict[str, Any], logger: PipelineLogger, tracker: PipelineTracker, toolchain: Dict[str, Any]):
        self.config = config.get("step02_sparse_sfm", {})
        self.toolchain = toolchain
        self.logger = logger
        self.tracker = tracker
        self.colmap_exe = self._resolve_colmap_path()

    def _resolve_colmap_path(self) -> Optional[str]:
        """Resolves the COLMAP executable across native tools, PATH, and WSL."""
        candidate = self.toolchain.get("colmap_bin", "tools/colmap/COLMAP.bat")
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
        
        # Check tools/colmap/COLMAP.bat or bin/colmap.exe relative to repo root
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        local_candidates = [
            os.path.join(root_dir, "tools", "bin", "colmap.exe"),
            os.path.join(root_dir, "tools", "COLMAP.bat"),
            os.path.join(root_dir, "tools", "colmap", "COLMAP.bat"),
            os.path.join(root_dir, "tools", "colmap", "bin", "colmap.exe"),
            os.path.join(root_dir, "tools", "colmap", "colmap.exe"),
        ]
        for c in local_candidates:
            if os.path.exists(c):
                return c

        # Check system PATH
        which_colmap = shutil.which("colmap") or shutil.which("COLMAP")
        if which_colmap:
            return which_colmap

        # Check WSL mode
        if self.toolchain.get("mode") in ("wsl", "auto"):
            try:
                res = subprocess.run(["wsl", "which", "colmap"], capture_output=True, text=True)
                if res.returncode == 0 and res.stdout.strip():
                    return "wsl colmap"
            except Exception:
                pass

        return None

    def _check_cuda_available(self) -> bool:
        """Determines if NVIDIA CUDA is available on this system."""
        if shutil.which("nvidia-smi"):
            try:
                res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                if res.returncode == 0:
                    return True
            except Exception:
                pass
        return False

    def _query_colmap_help(self, subcommand: str) -> str:
        """Runs colmap <subcommand> --help dynamically on this machine and caches output."""
        if not hasattr(self, "_help_cache"):
            self._help_cache = {}
        if subcommand in self._help_cache:
            return self._help_cache[subcommand]

        if not self.colmap_exe:
            return ""

        if self.colmap_exe.startswith("wsl"):
            cmd = ["wsl", "colmap", subcommand, "--help"]
        elif self.colmap_exe.endswith(".bat"):
            cmd = [self.colmap_exe, subcommand, "--help"]
        else:
            cmd = [self.colmap_exe, subcommand, "--help"]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            out = (res.stdout or "") + "\n" + (res.stderr or "")
            self._help_cache[subcommand] = out
            return out
        except Exception:
            return ""

    def run_sfm(
        self,
        frames: List[Dict[str, Any]],
        workspace_dir: str,
        intrinsics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes sparse reconstruction.
        Returns:
            Dict containing:
                - 'sparse_dir': path to aligned sparse model folder
                - 'database_path': path to COLMAP SQLite db
                - 'camera_poses': Dict[str, Dict] with camera centers & rotations
                - 'condition_ratio': float
                - 'mean_reprojection_error': float
        """
        self.logger.stage_header(2, "Sparse Reconstruction (COLMAP SfM)")
        
        if not self.colmap_exe:
            err_msg = (
                "COLMAP executable could not be located!\n"
                "Please ensure COLMAP is installed:\n"
                "  - Native Windows: run python scripts/download_colmap.py to install to tools/colmap/\n"
                "  - WSL2: install via `sudo apt-get install colmap` and configure toolchain.mode: 'wsl'\n"
                "Refer to README.md 'Environment & Toolchain Setup' for instructions."
            )
            self.logger.error(err_msg)
            raise RuntimeError(err_msg)

        self.logger.info(f"Using COLMAP binary at: {self.colmap_exe}")

        sfm_dir = os.path.join(workspace_dir, "sparse_sfm")
        images_dir = os.path.join(workspace_dir, "input_frames")
        os.makedirs(sfm_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        db_path = os.path.join(sfm_dir, "database.db")
        if os.path.exists(db_path):
            os.remove(db_path)

        # Ensure frames are accessible in images_dir (symlink or copy if needed)
        for f_info in frames:
            dest = os.path.join(images_dir, f_info["filename"])
            if not os.path.exists(dest):
                try:
                    os.link(f_info["path"], dest)
                except Exception:
                    shutil.copy2(f_info["path"], dest)

        has_cuda = self._check_cuda_available()

        # 1. Feature Extractor
        camera_model = self.config.get("camera_model", "SIMPLE_RADIAL")
        single_camera = "1" if self.config.get("single_camera", True) else "0"

        feat_cmd = [
            "feature_extractor",
            "--database_path", db_path,
            "--image_path", images_dir,
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", single_camera
        ]
        if intrinsics and "params" in intrinsics:
            params_str = ",".join(str(p) for p in intrinsics["params"])
            feat_cmd.extend(["--ImageReader.camera_params", params_str])

        # Dynamically inspect feature_extractor --help output on this machine
        feat_help = self._query_colmap_help("feature_extractor")
        if not has_cuda:
            if "--FeatureExtraction.use_gpu" in feat_help:
                feat_cmd.extend(["--FeatureExtraction.use_gpu", "0"])
            elif "--SiftExtraction.use_gpu" in feat_help:
                feat_cmd.extend(["--SiftExtraction.use_gpu", "0"])
            elif "--use_gpu" in feat_help:
                feat_cmd.extend(["--use_gpu", "0"])

        max_img_sz = self.config.get("max_image_size", 1600)
        if max_img_sz:
            if "--FeatureExtraction.max_image_size" in feat_help:
                feat_cmd.extend(["--FeatureExtraction.max_image_size", str(max_img_sz)])
            elif "--SiftExtraction.max_image_size" in feat_help:
                feat_cmd.extend(["--SiftExtraction.max_image_size", str(max_img_sz)])
            elif "--max_image_size" in feat_help:
                feat_cmd.extend(["--max_image_size", str(max_img_sz)])

        self.logger.info("Extracting image features...")
        self._run_colmap(feat_cmd)

        # 2. Sequential Matcher with Loop Detection
        match_cmd = [
            "sequential_matcher",
            "--database_path", db_path,
            "--SequentialMatching.overlap", "10"
        ]

        # Dynamically inspect sequential_matcher --help output on this machine
        match_help = self._query_colmap_help("sequential_matcher")
        if not has_cuda:
            if "--FeatureMatching.use_gpu" in match_help:
                match_cmd.extend(["--FeatureMatching.use_gpu", "0"])
            elif "--SiftMatching.use_gpu" in match_help:
                match_cmd.extend(["--SiftMatching.use_gpu", "0"])
            elif "--use_gpu" in match_help:
                match_cmd.extend(["--use_gpu", "0"])

        # Resolve local vocabulary tree for loop detection
        vocab_path = self.toolchain.get("vocab_tree_path")
        if not vocab_path or not os.path.exists(vocab_path):
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            default_vocab = os.path.join(root_dir, "tools", "vocab_tree", "vocab_tree_faiss_flickr100K_words256K.bin")
            if os.path.exists(default_vocab):
                vocab_path = default_vocab

        loop_det = self.config.get("loop_detection", True)
        if loop_det and vocab_path and os.path.exists(vocab_path):
            if "--SequentialMatching.vocab_tree_path" in match_help:
                match_cmd.extend([
                    "--SequentialMatching.loop_detection", "1",
                    "--SequentialMatching.vocab_tree_path", os.path.abspath(vocab_path)
                ])
                self.logger.info(f"Performing sequential matching with loop detection via local vocab tree: '{vocab_path}'...")
            else:
                match_cmd.extend(["--SequentialMatching.loop_detection", "1"])
                self.logger.info("Performing sequential matching (loop detection enabled)...")
        else:
            match_cmd.extend(["--SequentialMatching.loop_detection", "0"])
            self.logger.info("Performing sequential matching (loop detection disabled, sequence overlap enabled)...")

        self._run_colmap(match_cmd)

        # 3. Mapper (Sparse Reconstruction)
        sparse_out = os.path.join(sfm_dir, "sparse_raw")
        os.makedirs(sparse_out, exist_ok=True)
        mapper_cmd = [
            "mapper",
            "--database_path", db_path,
            "--image_path", images_dir,
            "--output_path", sparse_out
        ]
        self.logger.info("Running bundle adjustment mapper...")
        self._run_colmap(mapper_cmd)

        model_dir = os.path.join(sparse_out, "0")
        if not os.path.exists(model_dir):
            subdirs = [os.path.join(sparse_out, d) for d in os.listdir(sparse_out) if os.path.isdir(os.path.join(sparse_out, d))]
            if subdirs:
                model_dir = subdirs[0]
            else:
                raise RuntimeError("COLMAP mapper failed to reconstruct any camera models!")

        # 4. Early GPS-based Model Alignment Prior
        aligned_model_dir = os.path.join(sfm_dir, "sparse_aligned")
        os.makedirs(aligned_model_dir, exist_ok=True)

        gps_ref_file = os.path.join(sfm_dir, "gps_priors.txt")
        num_gps_priors = self._write_gps_priors(frames, gps_ref_file)

        if num_gps_priors >= 3 and self.config.get("early_gps_alignment", True):
            self.logger.info(f"Applying early GPS geo-alignment using {num_gps_priors} camera priors...")
            align_cmd = [
                "model_aligner",
                "--input_path", model_dir,
                "--output_path", aligned_model_dir,
                "--ref_images_path", gps_ref_file,
                "--robust_alignment", "1",
                "--robust_alignment_max_error", "5.0"
            ]
            try:
                self._run_colmap(align_cmd)
                final_sparse_dir = aligned_model_dir
                self.logger.success("Early GPS model alignment completed successfully.")
            except Exception as e:
                self.logger.warning(f"Model aligner prior failed: {e}. Proceeding with unaligned sparse model.")
                final_sparse_dir = model_dir
        else:
            if num_gps_priors == 0:
                self.logger.info("No GPS data available — using image-only relative camera poses (no geo-alignment).")
            else:
                self.logger.info(f"Only {num_gps_priors} GPS priors (need >= 3). Skipping geo-alignment.")
            final_sparse_dir = model_dir

        # 5. Export model to TXT / PLY for inspection & read camera positions
        txt_model_dir = os.path.join(sfm_dir, "sparse_txt")
        os.makedirs(txt_model_dir, exist_ok=True)
        conv_cmd = [
            "model_converter",
            "--input_path", final_sparse_dir,
            "--output_path", txt_model_dir,
            "--output_type", "TXT"
        ]
        self._run_colmap(conv_cmd)

        # 6. Parse camera poses & points, compute conditioning and reprojection error
        camera_poses, points3d, mean_reproj = self._parse_sparse_txt_model(txt_model_dir)

        reg_count = len(camera_poses)
        total_frames = len(frames)
        reg_rate = (reg_count / total_frames * 100.0) if total_frames > 0 else 0.0
        self.tracker.metrics["registered_frames"] = reg_count
        self.tracker.metrics["registration_rate_pct"] = round(reg_rate, 1)
        self.tracker.metrics["sparse_points"] = len(points3d)
        self.tracker.metrics["mean_reprojection_error_px"] = round(mean_reproj, 3)

        if mean_reproj > 1.5:
            self.tracker.metrics["reprojection_error_flag"] = "WARNING_HIGH_ERROR (>1.5px)"
            self.logger.warning(f"Mean reprojection error is high: {mean_reproj:.2f}px (> 1.5px)!")
        else:
            self.logger.success(f"Sparse reconstruction registered {reg_count}/{total_frames} frames ({reg_rate:.1f}%). Mean reprojection error: {mean_reproj:.2f}px.")

        # 7. Check Geometric Conditioning of Camera Trajectory
        cond_ratio, cond_status, cond_warning = self._check_trajectory_conditioning(camera_poses)
        self.tracker.metrics["geometry_conditioning"] = {
            "condition_ratio": round(cond_ratio, 4) if cond_ratio is not None else None,
            "status": cond_status,
            "warning": cond_warning
        }
        if cond_status == "POORLY_CONDITIONED_NEAR_LINEAR":
            self.logger.warning(cond_warning)
        else:
            self.logger.success(f"Camera trajectory geometric conditioning: {cond_status} (eigenvalue ratio: {cond_ratio:.3f}).")

        return {
            "sparse_dir": final_sparse_dir,
            "txt_model_dir": txt_model_dir,
            "database_path": db_path,
            "camera_poses": camera_poses,
            "points3d": points3d,
            "condition_ratio": cond_ratio,
            "mean_reprojection_error": mean_reproj
        }

    def _run_colmap(self, args: List[str]):
        """Executes a COLMAP command line safely."""
        if self.colmap_exe.startswith("wsl"):
            cmd = ["wsl", "colmap"] + args
        elif self.colmap_exe.endswith(".bat"):
            # Windows batch wrapper
            cmd = [self.colmap_exe] + args
        else:
            cmd = [self.colmap_exe] + args

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            err = res.stderr.strip() or res.stdout.strip()
            raise RuntimeError(f"COLMAP command failed: {' '.join(cmd)}\nError: {err}")

    def _write_gps_priors(self, frames: List[Dict[str, Any]], out_path: str) -> int:
        """Writes COLMAP model_aligner ref_images_path text file."""
        lines = []
        for f in frames:
            if f.get("gps"):
                g = f["gps"]
                # COLMAP format: IMAGE_NAME X Y Z (or lat, lon, alt)
                lines.append(f"{f['filename']} {g['lat']:.8f} {g['lon']:.8f} {g['alt']:.3f}")
        if lines:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        return len(lines)

    def _parse_sparse_txt_model(self, txt_dir: str) -> Tuple[Dict[str, Dict], List[np.ndarray], float]:
        """Parses images.txt and points3D.txt from COLMAP export."""
        images_file = os.path.join(txt_dir, "images.txt")
        points_file = os.path.join(txt_dir, "points3D.txt")
        
        camera_poses = {}
        if os.path.exists(images_file):
            with open(images_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # Iterate using a step to skip the interleaved metadata lines
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith("#"):
                    i += 1
                    continue
                # Line 1: ID, QW, QX, QY, QZ, TX, TY, TZ, CAM_ID, NAME
                parts = line.split()
                if len(parts) >= 10:
                    image_id = int(parts[0])
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    name = parts[9]
                    R = self._quat_to_rot(qw, qx, qy, qz)
                    t = np.array([tx, ty, tz])
                    camera_poses[name] = {
                        "id": image_id, "R": R, "t": t, "center": -R.T @ t, "quat": [qw, qx, qy, qz]
                    }
                i += 2 # Skip the second line (points)
        
        points3d = []
        errors = []
        if os.path.exists(points_file):
            with open(points_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    parts = line.split()
                    if len(parts) >= 8:
                        points3d.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
                        errors.append(float(parts[7]))
        return camera_poses, points3d, float(np.mean(errors)) if errors else 0.0

    def _quat_to_rot(self, qw, qx, qy, qz):
        """Converts quaternion to 3x3 rotation matrix."""
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        return R

    def _check_trajectory_conditioning(self, camera_poses: Dict[str, Dict]) -> Tuple[Optional[float], str, Optional[str]]:
        """
        Computes the geometric conditioning of the camera centers trace via Principal Component Analysis (PCA).
        If the ratio of the smallest to largest eigenvalue is near zero (< threshold), the flight path
        is essentially a 1D straight line, which cannot mathematically constrain rotation about the flight axis.
        """
        if len(camera_poses) < 3:
            return None, "INSUFFICIENT_CAMERAS", "Fewer than 3 registered cameras to evaluate flight path conditioning."

        centers = np.array([pose["center"] for pose in camera_poses.values()])
        # Center the data
        centered = centers - np.mean(centers, axis=0)
        cov = np.cov(centered, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(np.maximum(eigenvalues, 1e-12))[::-1] # sorted descending: l1 >= l2 >= l3

        # Ratio of secondary to primary principal axis (testing 1D straight line vs 2D areal/loop coverage)
        condition_ratio = float(eigenvalues[1] / eigenvalues[0])
        ratio_3d = float(eigenvalues[2] / eigenvalues[0])
        thresh = self.config.get("conditioning_threshold", 0.05)

        if condition_ratio < thresh:
            warning = (
                f"Flight path is geometrically near-linear (lateral condition ratio: {condition_ratio:.4f} < {thresh}). "
                "In single-pass flight, a near-linear trajectory cannot fully constrain 3D rotation about the flight axis. "
                "Downstream Helmert georeferencing will flag this structural limitation."
            )
            return condition_ratio, "POORLY_CONDITIONED_NEAR_LINEAR", warning
        else:
            return condition_ratio, "WELL_CONDITIONED", None
