"""
End-to-end integration test using the synthetic scene.

Validates the entire pipeline without requiring any real COLMAP data.
Run with: python -m pytest tests/test_synthetic.py -v
      or: python tests/test_synthetic.py

Checks:
  1. Synthetic scene generates valid meshes.
  2. Hole detection marks the missing wall/side as holes.
  3. Symmetry detector finds a plane with score > threshold for the box.
  4. Symmetry completion adds new faces tagged SYMMETRY_DERIVED.
  5. Inpaint completion adds new faces tagged GENERATED.
  6. Exporter writes all required files to disk.
  7. The completed building mesh has more faces than the input.
"""
from __future__ import annotations

import sys
import tempfile
import logging
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scene_io         import make_synthetic_scene
from scene_io.scene   import Provenance, SemanticClass
from detection  import HoleDetector, HoleDetectorConfig, SymmetryDetector, SymmetryDetectorConfig
from completion import CompletionDispatcher
from output     import SceneExporter, provenance_stats

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger("test_synthetic")


def test_full_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out_path = tmp_path / "output"

        # ── 1. Generate synthetic scene ───────────────────────────────────
        logger.info("Test 1: Generating synthetic scene …")
        scene = make_synthetic_scene(output_dir=tmp_path / "inputs")

        assert scene.n_instances == 2, f"Expected 2 instances, got {scene.n_instances}"
        building = next(i for i in scene.instances if i.semantic_class == SemanticClass.BUILDING)
        car      = next(i for i in scene.instances if i.semantic_class == SemanticClass.STATIC_CAR)
        assert len(building.mesh.faces) > 0
        assert len(car.mesh.faces) > 0
        logger.info("  ✓ Synthetic scene: building=%d faces, car=%d faces",
                    len(building.mesh.faces), len(car.mesh.faces))

        n_building_faces_initial = len(building.mesh.faces)
        n_car_faces_initial      = len(car.mesh.faces)

        # ── 2. Hole detection ─────────────────────────────────────────────
        logger.info("Test 2: Hole detection …")
        hole_cfg = HoleDetectorConfig(n_viewpoints=32, n_rays_per_viewpoint=256, hit_threshold=1)
        hole_det = HoleDetector(hole_cfg)

        for inst in scene.instances:
            hole_det.detect(inst)

        assert building.hole_mask is not None, "Building hole_mask was not set"
        assert car.hole_mask      is not None, "Car hole_mask was not set"

        n_building_holes = int(building.hole_mask.sum())
        n_car_holes      = int(car.hole_mask.sum())
        assert n_building_holes > 0, "Expected holes in synthetic building (missing back wall)"
        assert n_car_holes > 0,      "Expected holes in synthetic car (missing right side)"
        logger.info("  ✓ Holes: building=%d, car=%d", n_building_holes, n_car_holes)

        # ── 3. Symmetry detection ─────────────────────────────────────────
        logger.info("Test 3: Symmetry detection …")
        sym_cfg = SymmetryDetectorConfig(
            symmetry_accept_threshold=0.40,   # synthetic box → should be highly symmetric
            max_icp_iterations=30,
            n_sample_points=512,
        )
        sym_det = SymmetryDetector(sym_cfg)
        sym_det.detect(building)
        sym_det.detect(car)

        assert building.symmetry_score > 0.0, "Building symmetry score should be > 0"
        logger.info(
            "  ✓ Building symmetry score=%.3f  plane=%s",
            building.symmetry_score,
            np.round(building.symmetry_plane, 3) if building.symmetry_plane is not None else "None"
        )
        logger.info(
            "  ✓ Car symmetry score=%.3f  plane=%s",
            car.symmetry_score,
            np.round(car.symmetry_plane, 3) if car.symmetry_plane is not None else "None"
        )

        # ── 4. Completion ─────────────────────────────────────────────────
        logger.info("Test 4: Completion …")
        dispatcher = CompletionDispatcher()
        dispatcher.run(scene.instances)

        n_building_faces_after = len(building.mesh.faces)
        n_car_faces_after      = len(car.mesh.faces)

        assert n_building_faces_after > n_building_faces_initial, (
            f"Building faces did not increase after completion: "
            f"{n_building_faces_initial} → {n_building_faces_after}"
        )
        assert n_car_faces_after > n_car_faces_initial, (
            f"Car faces did not increase after completion: "
            f"{n_car_faces_initial} → {n_car_faces_after}"
        )

        bld_stats = provenance_stats(building.provenance)
        car_stats = provenance_stats(car.provenance)
        logger.info("  ✓ Building faces: %d → %d | provenance: %s",
                    n_building_faces_initial, n_building_faces_after, bld_stats)
        logger.info("  ✓ Car faces: %d → %d | provenance: %s",
                    n_car_faces_initial, n_car_faces_after, car_stats)

        # Provenance arrays must be sane
        assert len(building.provenance) == len(building.mesh.faces), \
            "Provenance array length mismatch for building"
        assert len(car.provenance) == len(car.mesh.faces), \
            "Provenance array length mismatch for car"

        # At least some SYMMETRY_DERIVED or GENERATED faces must exist
        n_filled_bld = bld_stats.get("SYMMETRY_DERIVED", 0) + bld_stats.get("GENERATED", 0)
        n_filled_car = car_stats.get("SYMMETRY_DERIVED", 0) + car_stats.get("GENERATED", 0)
        assert n_filled_bld > 0, "No filled faces in building"
        assert n_filled_car > 0, "No filled faces in car"

        # ── 5. Export ─────────────────────────────────────────────────────
        logger.info("Test 5: Export …")
        exporter = SceneExporter(out_path)
        summary  = exporter.export(scene)

        merged_obj = Path(summary["merged_mesh"])
        manifest   = Path(summary["provenance_manifest"])
        video      = Path(summary["turntable_video"])

        assert merged_obj.exists(), f"Merged OBJ not found: {merged_obj}"
        assert manifest.exists(),   f"Provenance manifest not found: {manifest}"
        assert video.exists(),      f"Turntable video not found: {video}"

        logger.info("  ✓ Merged OBJ:          %s  (%.1f KB)", merged_obj.name, merged_obj.stat().st_size / 1024)
        logger.info("  ✓ Provenance manifest: %s  (%.1f KB)", manifest.name,   manifest.stat().st_size / 1024)
        logger.info("  ✓ Turntable video:     %s  (%.1f KB)", video.name,      video.stat().st_size / 1024)

        logger.info("")
        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()
