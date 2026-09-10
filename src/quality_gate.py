import cv2
import logging
import numpy as np
import av

logger = logging.getLogger("Stage1_QualityGate")

def calibrate_video_blur(video_path):
    """
    Scans the video (approx 1 frame per second) to find the median Laplacian variance.
    Returns a dynamic blur threshold based on this median.
    """
    logger.info(f"Calibrating dynamic blur threshold for {video_path}...")
    try:
        container = av.open(video_path)
    except Exception as e:
        logger.error(f"Failed to open video for calibration: {e}")
        return 50.0  # fallback
        
    stream = container.streams.video[0]
    fps = stream.average_rate
    if not fps or fps == 0:
        fps = 30
    
    variances = []
    frame_count = 0
    
    for frame in container.decode(video=0):
        if frame.is_corrupt:
            continue
        
        # Only sample roughly 1 frame per second to be extremely fast
        if frame_count % int(fps) == 0:
            gray = cv2.cvtColor(frame.to_ndarray(format='bgr24'), cv2.COLOR_BGR2GRAY)
            # Use a smaller proxy for fast variance calculation
            h, w = gray.shape
            gray_small = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            laplacian_var = cv2.Laplacian(gray_small, cv2.CV_64F).var()
            variances.append(laplacian_var)
            
        frame_count += 1
        
        # Cap calibration to max 60 samples (60 seconds) to save time
        if len(variances) >= 60:
            break
            
    container.close()
    
    if not variances:
        return 50.0
        
    median_var = float(np.median(variances))
    # Threshold is half of the median variance, floored at 10 to avoid accepting pure motion blur
    threshold = max(10.0, median_var * 0.5)
    logger.info(f"Calibration complete: Median variance = {median_var:.1f}, Threshold set to = {threshold:.1f}")
    return threshold

class QualityGate:
    def __init__(self, blur_threshold=None, video_path=None, target_overlap=0.85):
        if blur_threshold is not None:
            self.blur_threshold = blur_threshold
        elif video_path is not None:
            self.blur_threshold = calibrate_video_blur(video_path)
        else:
            self.blur_threshold = 50.0 # fallback
            
        self.exposure_clipping_threshold = 0.50 # 50% pixels clipped is extreme
        self.target_overlap = target_overlap
        
    def check_frame(self, proxy_img, coverage_state, inliers_ratio):
        """
        Checks frame quality. Returns (is_accepted, gate_reason, low_confidence)
        """
        gray = cv2.cvtColor(proxy_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Near duplicate check (ORB inlier ratio proxy)
        # We drop frames if they share more than 'target_overlap' inliers with the anchor frame.
        if inliers_ratio > self.target_overlap:
            return False, "duplicate", False
            
        # 2. Blur check
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < self.blur_threshold
        
        # 3. Exposure clipping check
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        clipped_pixels = float(hist[0].item()) + float(hist[255].item())
        total_pixels = gray.size
        is_clipped = (clipped_pixels / total_pixels) > self.exposure_clipping_threshold
        
        if is_blurry or is_clipped:
            # SALVAGE RULE: Check if it's the only evidence for this region
            if coverage_state.is_region_under_covered():
                logger.info(f"Salvaging frame (blur={is_blurry}, clipped={is_clipped}) due to under-covered region")
                return True, None, True
            else:
                reason = "blur" if is_blurry else "exposure"
                return False, reason, False
                
        return True, None, False
