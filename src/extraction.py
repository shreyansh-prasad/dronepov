import av
import cv2
import logging
import sys
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Stage0_Extraction")

def extract_frames(video_path, proxy_scale=0.5):
    """
    Extract frames and real PTS using PyAV.
    Returns a generator yielding:
      (frame_id, pts_ms, proxy_frame, full_frame)
    """
    try:
        container = av.open(video_path)
    except Exception as e:
        logger.error(f"Failed to open video {video_path}: {e}")
        return

    stream = container.streams.video[0]
    time_base = stream.time_base
    
    frame_id = 0
    prev_micro_proxy = None
    
    for frame in container.decode(video=0):
        try:
            if frame.is_corrupt:
                logger.warning(f"Skipping corrupt frame at {frame.pts}")
                continue
            
            # Convert PyAV VideoFrame to numpy array (OpenCV uses BGR)
            img = frame.to_ndarray(format='bgr24')
            
            # Micro-proxy for ultra-fast stationary hovering detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            micro = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            
            if prev_micro_proxy is not None:
                mse = np.mean((micro.astype("float") - prev_micro_proxy.astype("float")) ** 2)
                if mse < 5.0:  # Threshold for effectively identical/hovering frames
                    continue
            
            prev_micro_proxy = micro
            
            pts = frame.pts
            if pts is None:
                logger.warning(f"Frame {frame_id} missing PTS, skipping.")
                continue
                
            # time_base is a fraction, multiply to get seconds, then 1000 for ms
            pts_ms = float(pts * time_base * 1000)
            
            # Create downsampled proxy
            h, w = img.shape[:2]
            new_w, new_h = int(w * proxy_scale), int(h * proxy_scale)
            proxy_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            yield frame_id, pts_ms, proxy_img, img
            frame_id += 1
            
        except Exception as e:
            logger.error(f"Error decoding frame {frame_id}: {e}")
            continue

    container.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        logger.info(f"Extracting frames from {video_file}...")
        for fid, pts, proxy, full in extract_frames(video_file):
            print(f"Frame {fid}: PTS={pts:.2f}ms, proxy_shape={proxy.shape}")
            if fid >= 10:
                print("... truncated for test ...")
                break
    else:
        print("Usage: python extraction.py <video_file>")
