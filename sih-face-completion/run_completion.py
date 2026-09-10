"""
sih-face-completion — Entry point.

Usage (production — with COLMAP mesh):
    python run_completion.py \\
        --colmap-dir  path/to/colmap_reconstruction/ \\
        --manifest    path/to/output/manifest.json \\
        --output-dir  path/to/completion_output/

Usage (synthetic test — no mesh needed):
    python run_completion.py --synthetic --output-dir path/to/output/

All pipeline stages run in order:
  Stage 1: Load scene (COLMAP or synthetic)
  Stage 2: Hole detection (raycasting)
  Stage 3: Symmetry detection (PCA + ICP)
  Stage 4: Completion (symmetry mirror or inpainting)
  Stage 5: Export (OBJ + provenance JSON + turntable video)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Configure logging before importing pipeline modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_completion")

# Add sih-face-completion directory to path so subpackages resolve correctly
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from scene_io            import load_colmap_scene, make_synthetic_scene
from detection     import HoleDetector, HoleDetectorConfig, SymmetryDetector, SymmetryDetectorConfig
from completion    import CompletionDispatcher
from output        import SceneExporter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="sih-face-completion: Fill missing faces in COLMAP drone reconstructions."
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--colmap-dir",
        type=Path,
        help="Root of the COLMAP reconstruction directory "
             "(must contain sparse/cameras.txt + images.txt and dense/meshed-poisson.obj).",
    )
    mode.add_argument(
        "--synthetic",
        action="store_true",
        help="Run on a synthetic test scene (no COLMAP data required).",
    )

    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest.json from the frame-selection pipeline "
             "(required when using --colmap-dir).",
    )
    p.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory of source frame images (defaults to <colmap-dir>/images/).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write all output files.",
    )

    # Tuning knobs
    p.add_argument("--hole-viewpoints",    type=int,   default=64,   help="Hemisphere viewpoints for hole detection.")
    p.add_argument("--hole-threshold",     type=int,   default=1,    help="Min ray hits to consider a face observed.")
    p.add_argument("--sym-threshold",      type=float, default=0.60, help="ICP score threshold for symmetry acceptance.")
    p.add_argument("--sym-icp-iters",      type=int,   default=50,   help="Max ICP iterations.")
    p.add_argument("--sym-sample-pts",     type=int,   default=2048, help="Points sampled for symmetry detection.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()

    # ── Stage 1: Load scene ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 1: Loading scene")
    logger.info("=" * 60)

    if args.synthetic:
        logger.info("Running in SYNTHETIC mode.")
        scene = make_synthetic_scene(output_dir=args.output_dir / "synthetic_inputs")
    else:
        if args.manifest is None:
            logger.error("--manifest is required when using --colmap-dir.")
            sys.exit(1)
        scene = load_colmap_scene(
            colmap_dir=args.colmap_dir,
            manifest_path=args.manifest,
            image_dir=args.image_dir,
        )

    logger.info(
        "Scene loaded: %d instances ready for processing.",
        scene.n_instances,
    )

    # ── Stage 2: Hole detection ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 2: Hole detection (raycasting)")
    logger.info("=" * 60)

    hole_cfg = HoleDetectorConfig(
        n_viewpoints=args.hole_viewpoints,
        hit_threshold=args.hole_threshold,
    )
    hole_detector = HoleDetector(hole_cfg)

    for inst in scene.instances:
        hole_detector.detect(inst)

    needs = [i for i in scene.instances if i.hole_mask is not None and i.hole_mask.any()]
    logger.info("%d / %d instances have holes requiring completion.", len(needs), scene.n_instances)

    # ── Stage 3: Symmetry detection ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 3: Symmetry detection (PCA + ICP)")
    logger.info("=" * 60)

    sym_cfg = SymmetryDetectorConfig(
        symmetry_accept_threshold=args.sym_threshold,
        max_icp_iterations=args.sym_icp_iters,
        n_sample_points=args.sym_sample_pts,
    )
    sym_detector = SymmetryDetector(sym_cfg)

    for inst in needs:
        if inst.completion_strategy == "symmetry":
            sym_detector.detect(inst)

    # ── Stage 4: Completion ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 4: Completion")
    logger.info("=" * 60)

    dispatcher = CompletionDispatcher()
    dispatcher.run(needs)

    # ── Stage 5: Export ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 5: Export")
    logger.info("=" * 60)

    exporter = SceneExporter(args.output_dir)
    summary  = exporter.export(scene)

    elapsed = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f seconds.", elapsed)
    logger.info("Output directory: %s", args.output_dir.resolve())
    for key, val in summary.items():
        logger.info("  %-25s: %s", key, val)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
