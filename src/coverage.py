import cv2
import numpy as np
import logging

logger = logging.getLogger("Stage3_Coverage")

class CoverageState:
    def __init__(self, frame_width, frame_height, cell_size=32):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cell_size = cell_size # CALIBRATE ON REAL FOOTAGE
        
        # Grid dimensions
        self.grid_w = (frame_width + cell_size - 1) // cell_size
        self.grid_h = (frame_height + cell_size - 1) // cell_size
        
        # Continuous coverage value per cell
        self.coverage_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Homography tracking: map current frame to global frame (frame 0)
        self.current_to_global_H = np.eye(3, dtype=np.float32)
        
        self.anchor_keypoints = None
        self.anchor_descriptors = None
        
        # State for the currently processed frame (before commit)
        self.current_keypoints = None
        self.current_descriptors = None
        self.current_H = None
        
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def get_overlap_ratio(self, proxy_img):
        """
        Computes ORB features and matches them against the last accepted anchor frame.
        Returns homography inliers ratio (used to measure overlap).
        Does NOT update the anchor frame.
        """
        gray = cv2.cvtColor(proxy_img, cv2.COLOR_BGR2GRAY)
        kps, descs = self.orb.detectAndCompute(gray, None)
        
        self.current_keypoints = kps
        self.current_descriptors = descs
        self.current_H = None
        
        inliers_ratio = 0.0
        
        if self.anchor_descriptors is not None and descs is not None and len(descs) > 0 and len(self.anchor_descriptors) > 0:
            matches = self.matcher.match(descs, self.anchor_descriptors)
            if len(matches) >= 10:
                src_pts = np.float32([ kps[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ self.anchor_keypoints[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
                
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    self.current_H = H
                    if mask is not None:
                        inliers = np.sum(mask)
                        inliers_ratio = inliers / len(matches)
        else:
            # If no anchor exists, this is the first frame. Overlap is 0 so it gets accepted.
            inliers_ratio = 0.0
            
        return inliers_ratio

    def commit_frame(self):
        """
        Called when a frame passes the Quality Gate.
        Sets the current frame as the new anchor and updates homography and footprint.
        """
        if self.current_H is not None:
            self.current_to_global_H = self.current_to_global_H @ self.current_H
            
        self.anchor_keypoints = self.current_keypoints
        self.anchor_descriptors = self.current_descriptors
        
        self._add_footprint_to_grid()

    def _add_footprint_to_grid(self):
        h, w = self.frame_height, self.frame_width
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        global_corners = cv2.perspectiveTransform(corners, self.current_to_global_H)
        
        pts = global_corners.reshape(-1, 2)
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        
        min_gx = max(0, int(min_x // self.cell_size))
        max_gx = min(self.grid_w - 1, int(max_x // self.cell_size))
        min_gy = max(0, int(min_y // self.cell_size))
        max_gy = min(self.grid_h - 1, int(max_y // self.cell_size))
        
        if min_gx <= max_gx and min_gy <= max_gy:
            # Increment coverage for this region
            self.coverage_grid[min_gy:max_gy+1, min_gx:max_gx+1] += 1.0

    def is_region_under_covered(self):
        """ Check if current viewport falls on an under-covered region """
        h, w = self.frame_height, self.frame_width
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        global_corners = cv2.perspectiveTransform(corners, self.current_to_global_H)
        
        pts = global_corners.reshape(-1, 2)
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        
        min_gx = max(0, int(min_x // self.cell_size))
        max_gx = min(self.grid_w - 1, int(max_x // self.cell_size))
        min_gy = max(0, int(min_y // self.cell_size))
        max_gy = min(self.grid_h - 1, int(max_y // self.cell_size))
        
        if min_gx <= max_gx and min_gy <= max_gy:
            region = self.coverage_grid[min_gy:max_gy+1, min_gx:max_gx+1]
            if region.size == 0:
                return False
            avg_coverage = np.mean(region)
            return avg_coverage < 1.0
        return False

    def report_coverage(self):
        total_cells = self.grid_w * self.grid_h
        unknown = np.sum(self.coverage_grid == 0)
        weakly = np.sum((self.coverage_grid > 0) & (self.coverage_grid < 2.0))
        observed = np.sum(self.coverage_grid >= 2.0)
        
        return (observed / total_cells, weakly / total_cells, unknown / total_cells)
