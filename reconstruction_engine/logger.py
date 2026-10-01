import os
import sys
import time
import json
import logging
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

# ANSI color codes for rich CLI logging
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"

class PipelineLogger:
    """Standardized logger for UAV 3D Reconstruction Pipeline."""
    def __init__(self, name: str = "UAV-Reconstruction", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        if not self.logger.handlers:
            # Console Handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter(f"{CYAN}[%(asctime)s]{RESET} [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

            # Optional File Handler
            if log_file:
                os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
                fh = logging.FileHandler(log_file, encoding="utf-8")
                fh.setLevel(logging.DEBUG)
                fh_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
                fh.setFormatter(fh_formatter)
                self.logger.addHandler(fh)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(f"{YELLOW}{BOLD}WARNING: {msg}{RESET}")

    def error(self, msg: str):
        self.logger.error(f"{RED}{BOLD}ERROR: {msg}{RESET}")

    def success(self, msg: str):
        self.logger.info(f"{GREEN}{BOLD}SUCCESS: {msg}{RESET}")

    def stage_header(self, step_num: int, title: str):
        bar = "=" * 68
        self.logger.info(f"\n{MAGENTA}{BOLD}{bar}{RESET}")
        self.logger.info(f"{MAGENTA}{BOLD}>>> STAGE {step_num}: {title.upper()}{RESET}")
        self.logger.info(f"{MAGENTA}{BOLD}{bar}{RESET}")


class PipelineTracker:
    """Tracks metrics, timings, degraded mode flags, and accuracy indicators across all pipeline stages."""
    def __init__(self):
        self.stage_timings: Dict[str, float] = {}
        self.stage_status: Dict[str, str] = {}  # 'SUCCESS', 'DEGRADED', 'SKIPPED', 'FAILED'
        self.degraded_reasons: Dict[str, str] = {}
        
        # Summary statistics
        self.metrics: Dict[str, Any] = {
            "total_frames": 0,
            "registered_frames": 0,
            "registration_rate_pct": 0.0,
            "sparse_points": 0,
            "dense_points": 0,
            "mesh_faces_full": 0,
            "mesh_faces_viewer": 0,
            "mesh_vertices": 0,
            "mean_reprojection_error_px": None,
            "reprojection_error_flag": "NORMAL", # "NORMAL" or "WARNING_HIGH_ERROR (>1.5px)"
            "georef_rms_residual_m": None,
            "crs_epsg": None,
            "geometry_conditioning": {
                "condition_ratio": None,
                "status": "UNKNOWN", # "WELL_CONDITIONED", "POORLY_CONDITIONED_NEAR_LINEAR"
                "warning": None
            },
            "provenance_breakdown": {
                "multi_view_points_pct": 100.0,
                "ai_depth_points_pct": 0.0
            },
            "validation_measurement": None
        }

    @contextmanager
    def time_stage(self, stage_name: str, logger: Optional[PipelineLogger] = None):
        start_time = time.perf_counter()
        if logger:
            logger.info(f"Starting {stage_name}...")
        try:
            yield
            duration = time.perf_counter() - start_time
            self.stage_timings[stage_name] = round(duration, 3)
            if stage_name not in self.stage_status:
                self.stage_status[stage_name] = "SUCCESS"
            if logger:
                status_color = YELLOW if self.stage_status.get(stage_name) == "DEGRADED" else GREEN
                logger.info(f"{status_color}Completed {stage_name} in {duration:.2f}s [{self.stage_status[stage_name]}]{RESET}")
        except Exception as e:
            duration = time.perf_counter() - start_time
            self.stage_timings[stage_name] = round(duration, 3)
            self.stage_status[stage_name] = "FAILED"
            if logger:
                logger.error(f"Failed {stage_name} after {duration:.2f}s: {e}")
            raise e

    def mark_degraded(self, stage_name: str, reason: str, logger: Optional[PipelineLogger] = None):
        self.stage_status[stage_name] = "DEGRADED"
        self.degraded_reasons[stage_name] = reason
        if logger:
            logger.warning(f"Stage '{stage_name}' running in DEGRADED mode: {reason}")

    def to_report_dict(self) -> Dict[str, Any]:
        total_time = sum(self.stage_timings.values())
        return {
            "summary": {
                "pipeline_version": "1.0.0",
                "total_wall_clock_time_sec": round(total_time, 2),
                "degraded_stages_count": sum(1 for s in self.stage_status.values() if s == "DEGRADED"),
                "is_fully_degraded": all(s == "DEGRADED" for s in self.stage_status.values())
            },
            "metrics": self.metrics,
            "stage_timings_sec": self.stage_timings,
            "stage_status": self.stage_status,
            "degraded_reasons": self.degraded_reasons
        }
