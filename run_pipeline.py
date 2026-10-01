import os
import sys
import hashlib
import json
import argparse
import yaml
import shutil
from typing import Dict, Any
from PIL import Image

from reconstruction_engine.logger import PipelineLogger, PipelineTracker
from reconstruction_engine.step01_input_validator import InputValidator
from reconstruction_engine.step02_sparse_sfm import SparseSfMEngine
from reconstruction_engine.step03_dense_mvs import DenseMVSEngine
from reconstruction_engine.step05_pointcloud_cleanup import PointCloudCleanupEngine
from reconstruction_engine.step06_surface_meshing import SurfaceMeshingEngine
from reconstruction_engine.step07_mesh_refinement import MeshRefinementEngine
from reconstruction_engine.step08_texture_mapping import TextureMappingEngine
from reconstruction_engine.step09_confidence_tagging import ConfidenceTaggingEngine
from reconstruction_engine.step11_exporter_reporter import ExporterReporterEngine

# Import vocab tree fetch utility
from scripts.fetch_vocab_tree import download_vocab_tree


def check_environment(config: Dict[str, Any], logger: PipelineLogger) -> bool:
    """Verifies Python version, library imports, and COLMAP/OpenMVS toolchain resolution."""
    logger.info("=== Running Environment & Toolchain Diagnostic Check ===")
    all_ok = True

    # 1. Python version check
    py_ver = sys.version.split()[0]
    logger.info(f"Python Runtime: {py_ver} ({sys.executable})")
    if sys.version_info < (3, 10):
        logger.error("Python >= 3.10 is required.")
        all_ok = False
    elif sys.version_info >= (3, 13):
        logger.warning(
            f"Python {py_ver} detected. Note that Open3D wheels currently require Python <= 3.12. "
            "Ensure you are running inside the pinned .venv312 environment."
        )

    # 2. Check core python photogrammetry libraries
    libraries = [
        ("open3d", "Open3D"),
        ("pymeshlab", "PyMeshLab"),
        ("trimesh", "Trimesh"),
        ("xatlas", "xatlas"),
        ("laspy", "laspy"),
        ("pyproj", "pyproj"),
        ("cv2", "OpenCV"),
        ("scipy", "SciPy"),
        ("numpy", "NumPy"),
    ]
    for mod_name, disp_name in libraries:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Installed")
            logger.success(f"Library {disp_name}: OK ({ver})")
        except Exception as e:
            logger.error(f"Library {disp_name}: NOT FOUND or FAILED ({e})")
            all_ok = False

    # 3. Check COLMAP binary
    toolchain = config.get("toolchain", {})
    colmap_path = toolchain.get("colmap_bin", "tools/colmap/COLMAP.bat")
    root_dir = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        colmap_path,
        os.path.join(root_dir, "tools", "bin", "colmap.exe"),
        os.path.join(root_dir, "tools", "colmap", "COLMAP.bat"),
        shutil.which("colmap"),
    ]
    colmap_found = None
    for c in candidates:
        if c and os.path.exists(c):
            colmap_found = c
            break

    if colmap_found:
        logger.success(f"COLMAP Binary: FOUND at '{colmap_found}'")
    else:
        logger.warning("COLMAP Binary: NOT FOUND in default paths. Run scripts/download_colmap.py to install.")

    # 4. Check OpenMVS
    openmvs_dir = toolchain.get("openmvs_bin_dir", "")
    reconstruct_bin = shutil.which("ReconstructMesh") or (os.path.join(openmvs_dir, "ReconstructMesh.exe") if openmvs_dir else None)
    if reconstruct_bin and os.path.exists(reconstruct_bin):
        logger.success(f"OpenMVS Binary: FOUND at '{reconstruct_bin}'")
    else:
        logger.info("OpenMVS Binary: Not found (Pipeline will utilize Open3D Poisson & xatlas fallback).")

    # 5. Check CUDA GPU
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        logger.success("NVIDIA CUDA GPU: AVAILABLE (GPU-accelerated Dense MVS enabled)")
    else:
        logger.info("NVIDIA CUDA GPU: NOT DETECTED (Pipeline will run in CPU mode)")

    logger.info("=== Diagnostic Check Completed ===")
    return all_ok


def _compute_dataset_fingerprint(input_dir: str) -> str:
    """Computes a deterministic fingerprint of the image set for safe resume.
    
    The fingerprint is based on:
    - sorted list of valid image filenames (excluding ._ files)
    - file sizes
    - total image count
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    entries = []
    for fname in sorted(os.listdir(input_dir)):
        if fname.startswith("._"):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in valid_extensions:
            continue
        fpath = os.path.join(input_dir, fname)
        if os.path.isfile(fpath):
            fsize = os.path.getsize(fpath)
            entries.append(f"{fname}:{fsize}")
    
    manifest = "\n".join(entries)
    fingerprint = hashlib.sha256(manifest.encode("utf-8")).hexdigest()[:16]
    return fingerprint


def _check_resume_safe(workspace_dir: str, current_fingerprint: str, logger: PipelineLogger) -> bool:
    """Checks if the previous reconstruction matches the current dataset.
    Returns True if safe to resume, False if workspace should be cleared."""
    manifest_path = os.path.join(workspace_dir, "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        return False  # No previous manifest — not safe to assume match
    
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("fingerprint") == current_fingerprint:
            logger.info(f"Dataset fingerprint matches previous run ({current_fingerprint}). Resume is safe.")
            return True
        else:
            logger.warning(
                f"Dataset fingerprint MISMATCH: previous={prev.get('fingerprint')}, current={current_fingerprint}. "
                "Clearing old workspace to prevent stale data reuse."
            )
            return False
    except Exception:
        return False


def _save_dataset_manifest(workspace_dir: str, fingerprint: str, image_count: int, input_dir: str):
    """Saves dataset manifest for future resume safety checks."""
    manifest_path = os.path.join(workspace_dir, "dataset_manifest.json")
    manifest = {
        "fingerprint": fingerprint,
        "image_count": image_count,
        "input_dir": os.path.abspath(input_dir),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def run_pipeline(
    input_dir: str,
    output_dir: str,
    gps_path: str = None,
    intrinsics_path: str = None,
    gcp_path: str = None,
    masks_dir: str = None,
    val_measurement_path: str = None,
    config_path: str = None,
    force_cpu: bool = False,
    resume_sparse: bool = False,
    dense_backend: str = "mvs",
    resume_dense: bool = False,
):
    """Executes the image-only 3D reconstruction pipeline."""
    # Load configuration
    if not config_path:
        config_path = os.path.join(os.path.dirname(__file__), "config", "pipeline_config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if force_cpu:
        config.setdefault("step03_dense_mvs", {})["force_cpu_fallback"] = True
    if dense_backend:
        config.setdefault("step03_dense_mvs", {})["backend"] = dense_backend
    if resume_dense:
        config.setdefault("step03_dense_mvs", {})["resume_dense"] = True

    # Determine reconstruction mode
    reconstruction_mode = config.get("reconstruction_mode", "image_only")
    is_image_only = (reconstruction_mode == "image_only")
    has_gps = gps_path is not None
    has_gcp = gcp_path is not None

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "pipeline.log")
    logger = PipelineLogger("3D-Reconstruction", log_file=log_file)
    tracker = PipelineTracker()
    toolchain = config.get("toolchain", {})

    workspace_dir = os.path.join(output_dir, "intermediate_workspace")
    os.makedirs(workspace_dir, exist_ok=True)

    # Record reconstruction mode in tracker
    tracker.metrics["reconstruction_mode"] = reconstruction_mode
    tracker.metrics["external_gps_used"] = has_gps
    tracker.metrics["external_telemetry_used"] = has_gps
    tracker.metrics["gcp_used"] = has_gcp
    tracker.metrics["external_camera_pose_used"] = False
    tracker.metrics["camera_pose_source"] = "ESTIMATED_FROM_IMAGES_VIA_SfM" if is_image_only else "GPS_ALIGNED"

    if is_image_only:
        logger.info("=" * 60)
        logger.info("RECONSTRUCTION MODE: IMAGE-ONLY")
        logger.info("Camera poses will be estimated from images using SfM.")
        logger.info("No GPS, GCP, or telemetry required.")
        logger.info("=" * 60)
    else:
        logger.info("Starting Georeferenced 3D Reconstruction Pipeline...")

    # ---------------------------------------------------------------
    # Dataset fingerprinting for safe resume
    # ---------------------------------------------------------------
    dataset_fingerprint = _compute_dataset_fingerprint(input_dir)
    logger.info(f"Dataset fingerprint: {dataset_fingerprint}")

    if resume_sparse or resume_dense:
        if not _check_resume_safe(workspace_dir, dataset_fingerprint, logger):
            logger.warning("Clearing workspace due to dataset mismatch. Starting fresh reconstruction.")
            # Clear workspace subdirectories that contain reconstruction data
            for subdir in ["sparse_sfm", "dense_mvs", "meshing", "refined_mesh", "textured_mesh", "confidence_mesh"]:
                path = os.path.join(workspace_dir, subdir)
                if os.path.exists(path):
                    shutil.rmtree(path)
            resume_sparse = False
            resume_dense = False

    # ---------------------------------------------------------------------
    # Ensure COLMAP vocabulary tree is present before SfM (loop detection)
    # ---------------------------------------------------------------------
    vocab_path = toolchain.get("vocab_tree_path")
    if not vocab_path or not os.path.exists(vocab_path):
        logger.info("Vocabulary tree missing – downloading now via fetch_vocab_tree script.")
        try:
            download_vocab_tree()
            logger.success("Vocabulary tree downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to fetch vocabulary tree: {e}")
            # Continue – SfM will fall back to internal loop detection if possible

    # Stage 1: Input Validation
    val_engine = InputValidator(config, logger, tracker)
    with tracker.time_stage("step01_input_validation", logger):
        validated_input = val_engine.validate_and_ingest(
            input_dir=input_dir,
            gps_path=gps_path,
            intrinsics_path=intrinsics_path,
            gcp_path=gcp_path,
            masks_dir=masks_dir,
        )

    frames = validated_input["frames"]
    intrinsics = validated_input["intrinsics"]
    gcps = validated_input["gcps"]

    # Save dataset manifest after successful validation
    _save_dataset_manifest(workspace_dir, dataset_fingerprint, len(frames), input_dir)

    # Stage 2: Copy input frames to workspace for downstream stages
    input_frames_dir = os.path.join(workspace_dir, "input_frames")
    os.makedirs(input_frames_dir, exist_ok=True)
    copied = 0
    for f in frames:
        src = f.get("path")
        if not src:
            logger.warning(f"Frame entry missing 'path': {f}")
            continue
        dst = os.path.join(input_frames_dir, f.get("filename", os.path.basename(src)))
        if os.path.exists(dst):
            continue
        shutil.copy2(src, dst)
        copied += 1

    # Verify copy results
    expected_count = len(frames)
    actual_files = [f for f in os.listdir(input_frames_dir) if os.path.isfile(os.path.join(input_frames_dir, f)) and not f.startswith("._")]
    if len(actual_files) != expected_count:
        logger.warning(f"Expected {expected_count} images in {input_frames_dir}, found {len(actual_files)}.")
    else:
        logger.info(f"All {expected_count} images present in workspace.")

    # Verify readability of images in workspace (report dimensions dynamically)
    for img_name in actual_files[:3]:  # Check first 3 as sample
        img_path = os.path.join(input_frames_dir, img_name)
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                logger.info(f"Sample image {img_name}: {w}x{h} px")
        except Exception as e:
            logger.error(f"Failed to open image {img_name}: {e}")

    if copied > 0:
        logger.info(f"Copied {copied}/{len(frames)} input frames to workspace for dense reconstruction.")

    # Stage 2: Sparse SfM (COLMAP Sequential Matching + Loop Detection)
    sfm_engine = SparseSfMEngine(config, logger, tracker, toolchain)

    # Determine if we can resume from existing sparse reconstruction
    sparse_raw_dir = os.path.join(workspace_dir, "sparse_sfm", "sparse_raw", "0")
    sparse_txt_dir = os.path.join(workspace_dir, "sparse_sfm", "sparse_txt")
    cameras_txt = os.path.join(sparse_txt_dir, "cameras.txt")
    images_txt = os.path.join(sparse_txt_dir, "images.txt")
    points3d_txt = os.path.join(sparse_txt_dir, "points3D.txt")
    
    def _file_ok(p):
        return os.path.isfile(p) and os.path.getsize(p) > 0
    
    if resume_sparse and os.path.isdir(sparse_raw_dir) and os.path.isdir(sparse_txt_dir) and _file_ok(cameras_txt) and _file_ok(images_txt) and _file_ok(points3d_txt):
        logger.info("Existing valid sparse reconstruction detected – resuming without re-running COLMAP.")
        # Parse existing model to obtain equivalent results dict
        camera_poses, points3d, reproj_err = sfm_engine._parse_sparse_txt_model(sparse_txt_dir)
        tracker.metrics["total_frames"] = len(frames)
        tracker.metrics["registered_frames"] = len(camera_poses)
        tracker.metrics["registration_rate_pct"] = round((len(camera_poses) / len(frames)) * 100.0, 1)
        tracker.metrics["sparse_points"] = len(points3d)
        tracker.metrics["mean_reprojection_error_px"] = round(reproj_err, 4)
        tracker.metrics["reprojection_error_flag"] = "NORMAL" if reproj_err <= 1.5 else "HIGH"
        tracker.stage_status["step02_sparse_sfm"] = "SUCCESS"
        tracker.stage_timings["step02_sparse_sfm"] = 0.0
        sfm_results = {
            "camera_poses": camera_poses,
            "sparse_dir": sparse_raw_dir,
            "sparse_txt_dir": sparse_txt_dir,
            "database_path": os.path.join(workspace_dir, "sparse_sfm", "database.db"),
            "points3d": points3d,
            "mean_reprojection_error": reproj_err,
        }
    else:
        if resume_sparse:
            logger.warning("Existing sparse reconstruction incomplete or invalid – running SfM normally.")
        with tracker.time_stage("step02_sparse_sfm", logger):
            sfm_results = sfm_engine.run_sfm(
                frames=frames,
                workspace_dir=workspace_dir,
                intrinsics=intrinsics,
            )

    camera_poses = sfm_results["camera_poses"]
    sparse_dir = sfm_results["sparse_dir"]

    # ---- Sparse Reconstruction Quality Check ----
    reg_count = tracker.metrics.get("registered_frames", len(camera_poses))
    total_count = tracker.metrics.get("total_frames", len(frames))
    reg_pct = tracker.metrics.get("registration_rate_pct", 0)
    sparse_pts = tracker.metrics.get("sparse_points", 0)
    reproj_err = tracker.metrics.get("mean_reprojection_error_px", 0)

    logger.info("=" * 60)
    logger.info("SPARSE RECONSTRUCTION QUALITY REPORT")
    logger.info(f"  Images discovered:       {total_count}")
    logger.info(f"  Images registered:       {reg_count}")
    logger.info(f"  Registration percentage: {reg_pct}%")
    logger.info(f"  Sparse 3D points:        {sparse_pts}")
    logger.info(f"  Mean reprojection error: {reproj_err} px")
    logger.info(f"  Camera model:            {config.get('step02_sparse_sfm', {}).get('camera_model', 'SIMPLE_RADIAL')}")
    logger.info("=" * 60)

    # Check if registration is sufficient for dense reconstruction
    if reg_count < 3:
        logger.error(
            f"FATAL: Only {reg_count} images registered. At least 3 are required for dense reconstruction. "
            "This usually means insufficient overlap between images. "
            "Try providing more overlapping images of the scene."
        )
        raise RuntimeError(f"Insufficient image registration ({reg_count}/{total_count}) for dense reconstruction.")

    if reg_pct < 20.0:
        logger.warning(
            f"WARNING: Only {reg_pct}% of images registered. Dense reconstruction may have poor coverage. "
            "Consider checking image overlap and quality."
        )

    # Stage 3: Dense MVS Reconstruction (NO AI depth in image-only mode)
    dense_engine = DenseMVSEngine(config, logger, tracker, toolchain)

    # In image-only mode, do NOT pass ai_depth_engine
    ai_depth_engine = None
    if not is_image_only and config.get("step04_ai_depth", {}).get("enabled", False):
        from reconstruction_engine.step04_ai_depth_fusion import AIDepthFusionEngine
        ai_depth_engine = AIDepthFusionEngine(config, logger, tracker)

    with tracker.time_stage("step03_dense_mvs", logger):
        dense_results = dense_engine.run_dense(
            frames=frames,
            sparse_dir=sparse_dir,
            workspace_dir=workspace_dir,
            ai_depth_engine=ai_depth_engine,
        )

    dense_cloud_path = dense_results["dense_cloud_path"]
    provenance_path = dense_results["provenance_path"]

    # Stage 5: Point Cloud Cleanup & Normal Estimation
    cleanup_engine = PointCloudCleanupEngine(config, logger, tracker)
    cleaned_ply = os.path.join(workspace_dir, "pointcloud_cleaned.ply")
    with tracker.time_stage("step05_pointcloud_cleanup", logger):
        pcd_clean, provenance_clean, clean_prov_path = cleanup_engine.cleanup_pointcloud(
            dense_ply_path=dense_cloud_path,
            provenance_path=provenance_path,
            output_ply_path=cleaned_ply,
        )

    # Stage 6: Surface Reconstruction (Meshing)
    meshing_engine = SurfaceMeshingEngine(config, logger, tracker, toolchain)
    with tracker.time_stage("step06_surface_meshing", logger):
        mesh_raw, densities, vert_provenance, mesh_raw_path = meshing_engine.reconstruct_surface(
            pcd=pcd_clean,
            provenance=provenance_clean,
            workspace_dir=workspace_dir,
        )

    # Stage 7: Mesh Refinement & Topology Cleanup
    refine_engine = MeshRefinementEngine(config, logger, tracker, toolchain)
    with tracker.time_stage("step07_mesh_refinement", logger):
        mesh_refined, densities_ref, vert_prov_ref, refined_ply = refine_engine.refine_mesh(
            mesh=mesh_raw,
            densities=densities,
            provenance=vert_provenance,
            workspace_dir=workspace_dir,
        )

    # Stage 8: UV Texture Mapping
    texture_engine = TextureMappingEngine(config, logger, tracker, toolchain)
    with tracker.time_stage("step08_texture_mapping", logger):
        texture_artifacts = texture_engine.texture_mesh(
            mesh=mesh_refined,
            frames=frames,
            camera_poses=camera_poses,
            workspace_dir=workspace_dir,
            output_prefix="mesh_full",
        )

    # Stage 9: Confidence Tagging (Multi-view provenance + observation geometry)
    confidence_engine = ConfidenceTaggingEngine(config, logger, tracker)
    with tracker.time_stage("step09_confidence_tagging", logger):
        confidence_artifacts = confidence_engine.compute_and_export_confidence(
            mesh=mesh_refined,
            densities=densities_ref,
            provenance=vert_prov_ref,
            camera_poses=camera_poses,
            workspace_dir=workspace_dir,
            output_prefix="mesh_confidence",
        )

    # Stage 10: Georeferencing (SKIPPED in image-only mode unless GPS data is available)
    georef_results = {
        "sim_transform": None,
        "rms_residual_m": 0.0,
        "crs_epsg": "LOCAL_RELATIVE",
        "conditioning_status": "IMAGE_ONLY_NO_GEOREF",
        "conditioning_warning": None,
    }

    georef_enabled = config.get("step10_georeferencing", {}).get("enabled", False)
    if georef_enabled and has_gps:
        from reconstruction_engine.step10_georeferencing import GeoreferencingEngine
        georef_engine = GeoreferencingEngine(config, logger, tracker)
        with tracker.time_stage("step10_georeferencing", logger):
            georef_results = georef_engine.georeference_scene(
                camera_poses=camera_poses,
                frames=frames,
                gcps=gcps,
                pcd=pcd_clean,
                mesh=mesh_refined,
                workspace_dir=workspace_dir,
            )
    else:
        if is_image_only:
            logger.info("Georeferencing SKIPPED (image-only mode — no GPS/GCP data required).")
        else:
            logger.info("Georeferencing SKIPPED (no GPS data available).")
        tracker.stage_status["step10_georeferencing"] = "SKIPPED"
        tracker.stage_timings["step10_georeferencing"] = 0.0

    # Stage 11: Multi-Format Deliverables Export & Report
    export_engine = ExporterReporterEngine(config, logger, tracker)
    with tracker.time_stage("step11_export_and_report", logger):
        exported_files = export_engine.export_all(
            pcd=pcd_clean,
            mesh_full=mesh_refined,
            texture_artifacts=texture_artifacts,
            confidence_artifacts=confidence_artifacts,
            georef_results=georef_results,
            output_dir=output_dir,
            val_measurement_path=val_measurement_path,
            densities=densities_ref,
            provenance=vert_prov_ref,
        )

    logger.info("\n============================================================")
    logger.success(f"Pipeline finished successfully! Outputs written to '{output_dir}'.")
    if is_image_only:
        logger.info("Mode: IMAGE-ONLY (no GPS/GCP/telemetry used)")
        logger.info("Camera poses: ESTIMATED FROM IMAGES")
        logger.info("Absolute scale: NOT GUARANTEED (relative reconstruction)")
    logger.info("============================================================\n")
    return exported_files


def main():
    parser = argparse.ArgumentParser(
        description="Image-Only 3D Mesh Reconstruction Engine — Generate a textured 3D mesh from overlapping photographs"
    )
    parser.add_argument("--input_dir", type=str, help="Directory containing overlapping photographs/images")
    parser.add_argument("--output_dir", type=str, default="output_image_only", help="Output directory for 3D deliverables")
    parser.add_argument("--gps", type=str, default=None, help="(Optional) Path to GPS telemetry file (.srt, .csv, .json)")
    parser.add_argument("--intrinsics", type=str, default=None, help="(Optional) Path to camera intrinsics JSON")
    parser.add_argument("--gcp", type=str, default=None, help="(Optional) Path to Ground Control Points CSV")
    parser.add_argument("--masks_dir", type=str, default=None, help="(Optional) Directory containing dynamic-object masks")
    parser.add_argument("--val_measurement", type=str, default=None, help="(Optional) Path to ground truth measurement JSON")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline_config.yaml")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU fallback mode")
    parser.add_argument("--check_env", action="store_true", help="Run environment & toolchain diagnostic check and exit")
    parser.add_argument("--resume_sparse", action="store_true", help="Resume from an existing valid COLMAP sparse reconstruction if present")
    parser.add_argument("--dense_backend", type=str, choices=["mvs", "ai_depth", "auto"], default="mvs", help="Dense reconstruction backend: 'mvs' (OpenMVS, default), 'ai_depth' (fallback), or 'auto'")
    parser.add_argument("--resume_dense", action="store_true", help="Resume from an existing valid dense MVS reconstruction if present")

    args = parser.parse_args()

    config_path = args.config or os.path.join(os.path.dirname(__file__), "config", "pipeline_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger = PipelineLogger("3D-Reconstruction")

    if args.check_env:
        ok = check_environment(config, logger)
        sys.exit(0 if ok else 1)

    if not args.input_dir:
        parser.error("--input_dir is required when running the reconstruction pipeline.")

    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gps_path=args.gps,
        intrinsics_path=args.intrinsics,
        gcp_path=args.gcp,
        masks_dir=args.masks_dir,
        val_measurement_path=args.val_measurement,
        config_path=args.config,
        force_cpu=args.force_cpu,
        resume_sparse=args.resume_sparse,
        dense_backend=args.dense_backend,
        resume_dense=args.resume_dense,
    )

if __name__ == "__main__":
    main()
