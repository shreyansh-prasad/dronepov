import os
import pytest
from reconstruction_engine.step04_ai_depth_fusion import AIDepthFusionEngine
from reconstruction_engine.logger import PipelineLogger, PipelineTracker

def test_read_colmap_dense_cameras():
    config = {}
    logger = PipelineLogger()
    tracker = PipelineTracker()
    engine = AIDepthFusionEngine(config, logger, tracker)
    dense_dir = os.path.abspath(os.path.join(os.getcwd(), 'output', 'intermediate_workspace', 'dense_mvs'))
    assert os.path.isdir(dense_dir), f"Dense directory not found: {dense_dir}"
    cameras, images = engine._read_colmap_dense_cameras(dense_dir)
    assert isinstance(cameras, dict) and isinstance(images, dict)
    assert len(cameras) > 0, "No cameras parsed"
    assert len(images) > 0, "No images parsed"
    cam = next(iter(cameras.values()))
    for key in ("fx", "fy", "cx", "cy", "w", "h"):
        assert key in cam, f"Camera missing key {key}"
    logger.info(f"Parsed {len(cameras)} cameras and {len(images)} images from COLMAP metadata.")
