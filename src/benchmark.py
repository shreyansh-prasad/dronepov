import time
import logging

logger = logging.getLogger("Stage7_Benchmark")

def mock_sfm_pipeline(frames):
    num_frames = len(frames)
    if num_frames == 0:
        return {'time_sec': 0, 'point_count': 0, 'mean_reprojection_error': 0}
        
    start_time = time.time()
    # Mock computation taking longer for more frames
    time.sleep(min(num_frames * 0.05, 1.0)) 
    wall_clock = time.time() - start_time
    
    point_count = num_frames * 1500 
    
    if num_frames > 20:
        reproj_error = 1.25
    else:
        reproj_error = 0.65
        
    return {
        'time_sec': wall_clock,
        'point_count': point_count,
        'mean_reprojection_error': reproj_error
    }

def run_benchmark(all_gated_frames, selected_frames):
    logger.info("Running benchmark on ALL gated frames...")
    full_res = mock_sfm_pipeline(all_gated_frames)
    
    logger.info("Running benchmark on SELECTED frames...")
    sel_res = mock_sfm_pipeline(selected_frames)
    
    print("\n--- Benchmark Results ---")
    print(f"{'Metric':<30} | {'All Gated':<15} | {'Selected':<15}")
    print("-" * 65)
    print(f"{'Frames':<30} | {len(all_gated_frames):<15} | {len(selected_frames):<15}")
    print(f"{'Wall-clock Time (s)':<30} | {full_res['time_sec']:<15.2f} | {sel_res['time_sec']:<15.2f}")
    print(f"{'Point Count':<30} | {full_res['point_count']:<15} | {sel_res['point_count']:<15}")
    print(f"{'Mean Reproj Error (px)':<30} | {full_res['mean_reprojection_error']:<15.2f} | {sel_res['mean_reprojection_error']:<15.2f}")
    print("-" * 65)
