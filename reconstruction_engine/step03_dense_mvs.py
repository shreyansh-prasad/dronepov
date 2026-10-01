import os
import sys
import shutil
import subprocess
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple
from reconstruction_engine.logger import PipelineLogger, PipelineTracker

class DenseMVSEngine:
    """
    Step 3: Dense Multi-View Stereo (MVS) Reconstruction.
    Runs COLMAP dense pipeline (undistorter -> patch_match_stereo -> stereo_fusion) if CUDA is present.
    If no CUDA GPU is available or forced, degrades gracefully to AI-depth/TSDF fusion.
    Dynamic-object masks are applied before stereo matching to prevent moving objects contaminating depth.
    """
    def __init__(self, config: Dict[str, Any], logger: PipelineLogger, tracker: PipelineTracker, toolchain: Dict[str, Any]):
        self.config = config.get("step03_dense_mvs", {})
        self.toolchain = toolchain
        self.logger = logger
        self.tracker = tracker
        self.colmap_exe = self._resolve_colmap()
        self.interface_colmap_exe, self.densify_exe = self._resolve_openmvs()

    def _resolve_colmap(self) -> str:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidates = [
            self.toolchain.get("colmap_bin"),
            os.path.join(root_dir, "tools", "bin", "colmap.exe"),
            os.path.join(root_dir, "tools", "colmap", "COLMAP.bat"),
            shutil.which("colmap")
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return c
        return "colmap"

    def _resolve_openmvs(self) -> Tuple[Optional[str], Optional[str]]:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        openmvs_dir = self.toolchain.get("openmvs_bin_dir", "")
        search_dirs = [
            openmvs_dir if openmvs_dir and os.path.isabs(openmvs_dir) else os.path.join(root_dir, openmvs_dir) if openmvs_dir else None,
            os.path.join(root_dir, "tools", "bin"),
            os.path.join(root_dir, "tools", "vc17", "x64", "Release")
        ]
        interface_bin = None
        densify_bin = None
        for d in search_dirs:
            if not d or not os.path.isdir(d):
                continue
            ic = os.path.join(d, "InterfaceCOLMAP.exe" if sys.platform == "win32" else "InterfaceCOLMAP")
            dp = os.path.join(d, "DensifyPointCloud.exe" if sys.platform == "win32" else "DensifyPointCloud")
            if not interface_bin and os.path.exists(ic):
                interface_bin = ic
            if not densify_bin and os.path.exists(dp):
                densify_bin = dp

        if not interface_bin:
            interface_bin = shutil.which("InterfaceCOLMAP")
        if not densify_bin:
            densify_bin = shutil.which("DensifyPointCloud")
        return interface_bin, densify_bin

    def run_dense(
        self,
        frames: List[Dict[str, Any]],
        sparse_dir: str,
        workspace_dir: str,
        ai_depth_engine: Any = None
    ) -> Dict[str, Any]:
        """
        Executes dense reconstruction.
        Returns:
            Dict containing:
                - 'dense_cloud_path': path to fused.ply
                - 'provenance_path': path to .npy array of point source tags ('mvs' vs 'ai_depth')
                - 'is_degraded': bool
                - 'dense_dir': path to dense workspace
        """
        self.logger.stage_header(3, "Dense Reconstruction (MVS / Volumetric Fusion)")

        dense_dir = os.path.join(workspace_dir, "dense_mvs")
        os.makedirs(dense_dir, exist_ok=True)
        images_dir = os.path.join(workspace_dir, "input_frames")

        # 1. Image Undistortion (CPU compatible)
        undist_sparse = os.path.join(dense_dir, "sparse")
        undist_images = os.path.join(dense_dir, "images")
        if os.path.isdir(undist_sparse) and os.path.isdir(undist_images) and len(os.listdir(undist_images)) >= len(frames):
            self.logger.info("Undistorted images already exist in dense workspace. Skipping COLMAP image undistorter.")
        else:
            self.logger.info("Running COLMAP image undistorter...")
            undistort_cmd = [
                self.colmap_exe, "image_undistorter",
                "--image_path", images_dir,
                "--input_path", sparse_dir,
                "--output_path", dense_dir,
                "--output_type", "COLMAP",
                "--max_image_size", str(self.config.get("max_image_size", 2000))
            ]
            self._exec(undistort_cmd)

        # 2. Apply Dynamic-Object Masks (gating matching BEFORE stereo depth estimation)
        if self.config.get("use_dynamic_masks", True):
            self._apply_dynamic_masks(frames, dense_dir)

        # 3. Check MVS backends
        backend = str(self.config.get("backend", "auto")).lower()
        has_cuda = self._check_cuda_available()
        force_cpu = self.config.get("force_cpu_fallback", False)
        has_openmvs = (self.interface_colmap_exe is not None and self.densify_exe is not None)

        fused_ply_path = os.path.join(dense_dir, "fused.ply")
        provenance_path = os.path.join(dense_dir, "provenance.npy")

        resume_dense = self.config.get("resume_dense", False)
        scene_dense_ply = os.path.join(dense_dir, "scene_dense.ply")

        if resume_dense and os.path.exists(scene_dense_ply) and self._count_ply_points(scene_dense_ply) > 1000:
            self.logger.info(f"Reusing existing valid OpenMVS dense reconstruction at '{scene_dense_ply}'.")
            shutil.copy2(scene_dense_ply, fused_ply_path)
            num_points = self._count_ply_points(fused_ply_path)
            provenance = np.array(["mvs"] * num_points, dtype=object)
            np.save(provenance_path, provenance)
            is_degraded = False
            self.logger.success(f"OpenMVS dense MVS verified: {num_points:,} real multi-view points loaded.")

        elif (backend == "mvs" or (backend == "auto" and has_openmvs)) and has_openmvs:
            # Primary True Multi-View Stereo Path: OpenMVS
            self.logger.info(f"Executing OpenMVS multi-view stereo densification ({self.densify_exe})...")
            dense_dir_abs = os.path.abspath(dense_dir)
            scene_mvs = os.path.join(dense_dir, "scene.mvs")
            scene_mvs_abs = os.path.abspath(scene_mvs)
            
            # Step 1: Export COLMAP undistorted workspace to scene.mvs
            if os.path.exists(scene_mvs_abs) and os.path.getsize(scene_mvs_abs) > 1000:
                self.logger.info(f"Reusing existing valid OpenMVS scene file at '{scene_mvs}'.")
            else:
                self.logger.info("Running InterfaceCOLMAP to import sparse model and undistorted images...")
                self._exec([self.interface_colmap_exe, "-i", dense_dir_abs, "-o", scene_mvs_abs, "-v", "3"], cwd=dense_dir)

            # Step 2: Run DensifyPointCloud (patch-match stereo across multi-views)
            self.logger.info("Running DensifyPointCloud across multi-view image pairs...")
            self._exec([self.densify_exe, scene_mvs_abs, "--resolution-level", "2", "--max-threads", "4", "-v", "3"], cwd=dense_dir)

            if os.path.exists(scene_dense_ply):
                shutil.copy2(scene_dense_ply, fused_ply_path)

            num_points = self._count_ply_points(fused_ply_path)
            provenance = np.array(["mvs"] * num_points, dtype=object)
            np.save(provenance_path, provenance)
            is_degraded = False
            self.logger.success(f"OpenMVS dense MVS completed: {num_points:,} real multi-view points generated.")

        elif has_cuda and not force_cpu and backend != "ai_depth":
            # Native GPU Path
            self.logger.info("CUDA GPU detected. Executing COLMAP patch_match_stereo...")
            pms_cmd = [
                self.colmap_exe, "patch_match_stereo",
                "--workspace_path", dense_dir,
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.window_radius", str(self.config.get("window_radius", 5)),
                "--PatchMatchStereo.num_iterations", str(self.config.get("num_iterations", 5)),
                "--PatchMatchStereo.geom_consistency", "1" if self.config.get("geom_consistency", True) else "0",
                "--PatchMatchStereo.filter_min_ncc", str(self.config.get("filter_min_ncc", 0.1))
            ]
            self._exec(pms_cmd)

            self.logger.info("Executing COLMAP stereo_fusion...")
            fuse_cmd = [
                self.colmap_exe, "stereo_fusion",
                "--workspace_path", dense_dir,
                "--workspace_format", "COLMAP",
                "--output_path", fused_ply_path
            ]
            self._exec(fuse_cmd)

            num_points = self._count_ply_points(fused_ply_path)
            provenance = np.array(["mvs"] * num_points, dtype=object)
            np.save(provenance_path, provenance)
            is_degraded = False
            self.logger.success(f"Native MVS fusion completed: {num_points} dense points generated.")
        else:
            # AI-Depth Fallback Path
            reason = "Real MVS backend unavailable (no OpenMVS and no CUDA GPU). Running AI-depth fallback."
            self.tracker.mark_degraded("step03_dense_mvs", reason, self.logger)
            self.logger.warning("Executing fallback: AI-depth volumetric fusion with least-squares multi-view scale alignment.")

            if ai_depth_engine is None:
                from reconstruction_engine.step04_ai_depth_fusion import AIDepthFusionEngine
                ai_depth_engine = AIDepthFusionEngine(self.config, self.logger, self.tracker)

            fused_ply_path, provenance_path = ai_depth_engine.fuse_dense_pointcloud(
                dense_dir=dense_dir,
                frames=frames,
                output_ply=fused_ply_path
            )
            is_degraded = True

        self.tracker.metrics["dense_points"] = self._count_ply_points(fused_ply_path)

        return {
            "dense_cloud_path": fused_ply_path,
            "provenance_path": provenance_path,
            "is_degraded": is_degraded,
            "dense_dir": dense_dir
        }

    def _check_cuda_available(self) -> bool:
        """Determines if NVIDIA CUDA is available on this system."""
        # Check nvidia-smi
        if shutil.which("nvidia-smi"):
            try:
                res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                if res.returncode == 0:
                    return True
            except Exception:
                pass
        return False

    def _apply_dynamic_masks(self, frames: List[Dict[str, Any]], dense_dir: str):
        """
        Applies dynamic-object masks to undistorted images.
        Masked pixels (vehicles, pedestrians) are zeroed out so they cannot form false stereo matches.
        """
        undist_images_dir = os.path.join(dense_dir, "images")
        if not os.path.exists(undist_images_dir):
            return

        applied_count = 0
        for f in frames:
            mask_path = f.get("mask_path")
            if not mask_path or not os.path.exists(mask_path):
                continue
            
            img_path = os.path.join(undist_images_dir, f["filename"])
            if not os.path.exists(img_path):
                continue

            try:
                with Image.open(img_path) as img, Image.open(mask_path) as mask:
                    mask = mask.convert("L").resize(img.size, Image.Resampling.NEAREST)
                    img_arr = np.array(img)
                    mask_arr = np.array(mask)
                    # Masked pixels (moving objects: mask > 128) are zeroed out
                    img_arr[mask_arr > 128] = 0
                    Image.fromarray(img_arr).save(img_path)
                    applied_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to apply dynamic mask for {f['filename']}: {e}")

        if applied_count > 0:
            self.logger.success(f"Gated stereo matching with dynamic-object masks on {applied_count} frames.")

    def _exec(self, cmd: List[str], cwd: Optional[str] = None):
        process = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        lines = []
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                lines.append(line)
                clean = line.strip()
                if clean and any(k in clean.lower() for k in ["estimated", "fusing", "filtering", "loaded", "saved", "error", "warning"]):
                    self.logger.info(clean)
        process.wait()
        if process.returncode != 0:
            err = "".join(lines[-20:]).strip()
            raise RuntimeError(f"Dense MVS execution error: {' '.join(cmd)}\n{err}")

    def _count_ply_points(self, ply_path: str) -> int:
        if not os.path.exists(ply_path):
            return 0
        try:
            with open(ply_path, "rb") as f:
                header = b""
                while True:
                    line = f.readline()
                    header += line
                    if line.startswith(b"element vertex"):
                        parts = line.strip().split()
                        return int(parts[-1])
                    if line.startswith(b"end_header"):
                        break
        except Exception:
            pass
        return 0
