import os
import tempfile
import yaml
import pytest
import numpy as np
from PIL import Image

from reconstruction_engine.step01_input_validator import InputValidator
from reconstruction_engine.step11_exporter_reporter import ExporterReporterEngine
from reconstruction_engine.logger import PipelineLogger, PipelineTracker
from run_pipeline import _compute_dataset_fingerprint


def _create_dummy_image(filepath: str, width: int = 100, height: int = 100):
    img = Image.new("RGB", (width, height), color=(120, 150, 180))
    img.save(filepath, format="JPEG")


def test_resource_fork_filtering():
    """Verify that macOS resource fork '._' files are ignored during input validation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create normal images (min 3 frames required by default)
        img1 = os.path.join(tmp_dir, "frame_001.jpg")
        img2 = os.path.join(tmp_dir, "frame_002.jpg")
        img3 = os.path.join(tmp_dir, "frame_003.jpg")
        _create_dummy_image(img1)
        _create_dummy_image(img2)
        _create_dummy_image(img3)

        # Create macOS resource fork files
        dot_clean = os.path.join(tmp_dir, "._frame_001.jpg")
        with open(dot_clean, "wb") as f:
            f.write(b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X")

        dot_other = os.path.join(tmp_dir, "._another.jpg")
        with open(dot_other, "wb") as f:
            f.write(b"resource fork")

        engine = InputValidator({"step01_validation": {"require_gps": False}}, PipelineLogger("TestLogger"), PipelineTracker())
        result = engine.validate_and_ingest(tmp_dir)
        frames = result["frames"]

        assert len(frames) == 3
        basenames = [os.path.basename(f["path"]) for f in frames]
        assert "frame_001.jpg" in basenames
        assert "frame_002.jpg" in basenames
        assert "frame_003.jpg" in basenames
        assert "._frame_001.jpg" not in basenames
        assert "._another.jpg" not in basenames


def test_image_only_no_gps_acceptance():
    """Verify that input validator accepts images without any GPS EXIF tags when require_gps=False."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        img1 = os.path.join(tmp_dir, "no_gps_1.jpg")
        img2 = os.path.join(tmp_dir, "no_gps_2.jpg")
        img3 = os.path.join(tmp_dir, "no_gps_3.jpg")
        _create_dummy_image(img1)
        _create_dummy_image(img2)
        _create_dummy_image(img3)

        engine = InputValidator({"step01_validation": {"require_gps": False}}, PipelineLogger("TestLogger"), PipelineTracker())
        result = engine.validate_and_ingest(tmp_dir)
        frames = result["frames"]

        assert len(frames) == 3
        for f in frames:
            assert f["gps"] == {} or f["gps"] is None


def test_dataset_fingerprint_consistency():
    """Verify that dataset fingerprinting is deterministic and sensitive to image set changes."""
    with tempfile.TemporaryDirectory() as tmp_dir_a, tempfile.TemporaryDirectory() as tmp_dir_b:
        # Create identical sets in dir A and dir B
        for d in (tmp_dir_a, tmp_dir_b):
            _create_dummy_image(os.path.join(d, "img1.jpg"))
            _create_dummy_image(os.path.join(d, "img2.jpg"))

        fp_a = _compute_dataset_fingerprint(tmp_dir_a)
        fp_b = _compute_dataset_fingerprint(tmp_dir_b)
        assert fp_a == fp_b, "Identical datasets must produce identical fingerprints"
        assert len(fp_a) == 16, "Fingerprint should be 16 characters"

        # Add a 3rd image to dir A
        _create_dummy_image(os.path.join(tmp_dir_a, "img3.jpg"))
        fp_a_modified = _compute_dataset_fingerprint(tmp_dir_a)
        assert fp_a_modified != fp_b, "Adding an image must alter the dataset fingerprint"


def test_image_only_pipeline_config_defaults():
    """Verify pipeline_config.yaml has image-only defaults configured."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "pipeline_config.yaml")
    assert os.path.exists(config_path), f"Config file not found at {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("reconstruction_mode") == "image_only"
    assert cfg.get("step01_validation", {}).get("require_gps") is False
    assert cfg.get("step04_ai_depth", {}).get("enabled") is False
    assert cfg.get("step10_georeferencing", {}).get("enabled") is False


def test_image_only_report_generation():
    """Verify report.md and summary.html accurately identify image-only mode without false geospatial claims."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = PipelineLogger("TestLogger")
        tracker = PipelineTracker()
        tracker.metrics["reconstruction_mode"] = "image_only"
        tracker.metrics["gps_used"] = False
        tracker.metrics["total_frames"] = 100
        tracker.metrics["registered_frames"] = 96
        tracker.metrics["registration_rate_pct"] = 96.0
        tracker.metrics["dense_points"] = 250000
        tracker.metrics["mesh_vertices_full"] = 120000
        tracker.metrics["mesh_faces_full"] = 240000
        tracker.metrics["mesh_faces_decimated"] = 19200
        tracker.metrics["mean_reprojection_error_px"] = 0.58
        tracker.metrics["reprojection_error_flag"] = "NORMAL"

        engine = ExporterReporterEngine({}, logger, tracker)
        data = tracker.to_report_dict()

        md_path = os.path.join(tmp_dir, "report.md")
        html_path = os.path.join(tmp_dir, "summary.html")

        engine._generate_markdown_report(data, md_path)
        engine._generate_html_summary(data, html_path)

        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        assert "IMAGE-ONLY" in md_content
        assert "N/A (Image-Only Mode — No GPS/GCP Used)" in md_content
        assert "100 / 96 (96.0%)" in md_content
        assert "250,000 points" in md_content

        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        assert "IMAGE-ONLY" in html_content
        assert "SKIPPED (Image-Only Mode)" in html_content
        assert "96 / 100" in html_content
        assert "250,000" in html_content
