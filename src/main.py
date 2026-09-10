import os
import cv2
import numpy as np
import logging
import torch

from extraction import extract_frames
from quality_gate import QualityGate
from coverage import CoverageState
from segmentation import SegmentationAgent, AERIAL_CLASS_NAMES
from selection import select_frames
from manifest import generate_manifest
from benchmark import run_benchmark

import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

def process_video(video_path, budget_k=None):
    if not os.path.exists(video_path):
        logger.error(f"Video {video_path} not found. Please provide a valid path.")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join("output", video_name)
    
    # Clean output directory for a fresh run
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Configuration (CALIBRATE ON REAL FOOTAGE)
    BLUR_THRESHOLD = 50.0    # Laplacian variance below this = blurry
    CELL_SIZE = 32           # Coverage grid cell size in pixels
    PROCESSING_COST = 0.1
    TARGET_OVERLAP = 0.85    # Adaptive sampling: Drop frames if they overlap > 85% with the last anchor frame
    BUDGET_K = budget_k      # Set if explicitly provided via CLI, else defaults to len(candidates)
    
    # Init agents
    q_gate = QualityGate(video_path=video_path, target_overlap=TARGET_OVERLAP)
    seg_agent = SegmentationAgent()
    
    coverage_state = None
    all_gated = []
    candidates = []
    class_pixel_totals = {}
    total_frames_decoded = 0
    total_frames_rejected_blur = 0
    total_frames_rejected_duplicate = 0
    total_frames_rejected_exposure = 0
    total_frames_rejected_mission_val = 0
    total_frames_salvaged = 0
    video_duration_ms = 0.0
    
    logger.info("--- Starting Pipeline ---")
    
    for frame_id, pts, proxy, full in extract_frames(video_path):
        if coverage_state is None:
            h, w = proxy.shape[:2]
            coverage_state = CoverageState(w, h, cell_size=CELL_SIZE)
        
        total_frames_decoded += 1
        video_duration_ms = max(video_duration_ms, pts)
        
        # Calculate overlap against the last accepted anchor frame
        inliers_ratio = coverage_state.get_overlap_ratio(proxy)
        
        is_accepted, reason, low_conf = q_gate.check_frame(proxy, coverage_state, inliers_ratio)
        
        if is_accepted:
            if low_conf:
                total_frames_salvaged += 1
                logger.info(f"Frame {frame_id} SALVAGED at {pts:.0f}ms (low_conf=True)")
            else:
                logger.info(f"Frame {frame_id} accepted at {pts:.0f}ms")
            
            # Commit this frame as the new anchor and update footprint
            coverage_state.commit_frame()
            
            mask, dyn_mask, mission_val = seg_agent.segment(full)
            
            if mission_val < 0.05:
                logger.info(f"Frame {frame_id} pruned due to low mission value ({mission_val:.3f})")
                total_frames_rejected_mission_val += 1
                continue
            
            # --- ADAPTIVE SAMPLING ---
            # If the current frame contains vehicles or people, we increase the sampling rate 
            # (allow higher overlap) for the next frames so downstream trackers have more temporal data.
            has_dynamic_objects = np.any(np.isin(mask, [5, 6, 7])) # static_car, moving_car, human
            if has_dynamic_objects:
                q_gate.target_overlap = 0.96  # High framerate (accept frames with up to 96% overlap)
            else:
                q_gate.target_overlap = TARGET_OVERLAP  # Revert to base (e.g. 0.85)
                
            unique, counts = np.unique(mask, return_counts=True)
            hist = {}
            for k, v in zip(unique, counts):
                cls_name = AERIAL_CLASS_NAMES[k] if k < len(AERIAL_CLASS_NAMES) else str(k)
                hist[cls_name] = float(v) / mask.size
                class_pixel_totals[cls_name] = class_pixel_totals.get(cls_name, 0) + int(v)
                
            footprint = np.zeros_like(coverage_state.coverage_grid)
            h_f, w_f = proxy.shape[:2]
            corners = np.float32([[0,0],[w_f,0],[w_f,h_f],[0,h_f]]).reshape(-1,1,2)
            global_corners = cv2.perspectiveTransform(corners, coverage_state.current_to_global_H)
            pts_bbox = global_corners.reshape(-1, 2)
            min_x, min_y = np.min(pts_bbox, axis=0)
            max_x, max_y = np.max(pts_bbox, axis=0)
            min_gx = max(0, int(min_x // CELL_SIZE))
            max_gx = min(coverage_state.grid_w - 1, int(max_x // CELL_SIZE))
            min_gy = max(0, int(min_y // CELL_SIZE))
            max_gy = min(coverage_state.grid_h - 1, int(max_y // CELL_SIZE))
            
            if min_gx <= max_gx and min_gy <= max_gy:
                footprint[min_gy:max_gy+1, min_gx:max_gx+1] = 1.0

            cand_dict = {
                'id': frame_id,
                'pts': pts,
                'proxy': proxy,
                'full': full,
                'mask': mask,
                'low_confidence': low_conf,
                'semantic_histogram': hist,
                'mission_value': mission_val,
                'grid': footprint,
                'processing_cost': PROCESSING_COST
            }
            
            all_gated.append(cand_dict)
            candidates.append(cand_dict)
        else:
            if reason == 'duplicate':
                total_frames_rejected_duplicate += 1
            elif reason == 'blur':
                total_frames_rejected_blur += 1
            elif reason == 'exposure':
                total_frames_rejected_exposure += 1
            logger.debug(f"Frame {frame_id} rejected: {reason}")

    logger.info("--- Running Selection ---")
    # For high-accuracy photogrammetry/SfM, retain ALL quality-gated non-duplicate frames.
    # Quality-Gate already pruned identical/blurry frames (80%+ reduction).
    if BUDGET_K is None:
        BUDGET_K = len(candidates)
    logger.info(f"Frame budget K={BUDGET_K} (retaining all {len(candidates)} non-duplicate quality frames for max SfM accuracy)")
    
    selected = select_frames(candidates, k=BUDGET_K, processing_cost_per_frame=PROCESSING_COST)
    
    # Save mask overlays ONLY for the final selected frames
    logger.info(f"Saving output mask overlays for the {len(selected)} selected frames...")
    for f in selected:
        mask_path = os.path.join(out_dir, f"mask_{f['id']}.png")
        seg_agent.save_mask_overlay(f['full'], f['mask'], mask_path)
        f['mask_path'] = mask_path

    logger.info("")
    logger.info("========== PIPELINE SUMMARY ==========")
    logger.info(f"  Total frames decoded  : {total_frames_decoded}")
    logger.info(f"  Frames passed gate    : {len(all_gated)}")
    logger.info(f"  Frames rejected       : {total_frames_decoded - len(all_gated)}")
    logger.info(f"    - Duplicates        : {total_frames_rejected_duplicate}")
    logger.info(f"    - Blurry            : {total_frames_rejected_blur}")
    logger.info(f"    - Over-exposed      : {total_frames_rejected_exposure}")
    logger.info(f"    - Low Mission Val   : {total_frames_rejected_mission_val}")
    logger.info(f"  Frames salvaged       : {total_frames_salvaged}")
    logger.info(f"  Frames selected (k)   : {len(selected)} / {BUDGET_K}")
    logger.info("======================================")
    
    obs, weak, unk = coverage_state.report_coverage()
    report = {
        "video_path": video_path,
        "video_duration_s": round(video_duration_ms / 1000.0, 2),
        "total_frames_decoded": total_frames_decoded,
        "total_frames_passed_gate": len(all_gated),
        "total_frames_rejected": total_frames_decoded - len(all_gated),
        "rejected_breakdown": {
            "duplicates": total_frames_rejected_duplicate,
            "blurry": total_frames_rejected_blur,
            "over_exposed": total_frames_rejected_exposure,
            "low_mission_val": total_frames_rejected_mission_val
        },
        "total_frames_salvaged": total_frames_salvaged,
        "total_frames_selected": len(selected),
        "frame_budget_k": BUDGET_K,
        "coverage_by_status": {
            "observed_pct": float(obs),
            "weakly_observed_pct": float(weak),
            "unknown_pct": float(unk)
        },
        "class_pixel_totals": class_pixel_totals
    }
    
    generate_manifest(selected, report, os.path.join(out_dir, "manifest.json"), os.path.join(out_dir, "report.json"))
    run_benchmark(all_gated, selected)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/"
    custom_k = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_file = os.path.join(path, f)
                logger.info(f"Processing video in directory: {video_file}")
                process_video(video_file, budget_k=custom_k)
    else:
        process_video(path, budget_k=custom_k)

if __name__ == "__main__":
    main()
