import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM
from concurrent.futures import ThreadPoolExecutor

# YOLO class IDs corresponding to objects that can potentially move.
DYNAMIC_CLASSES = [0, 1, 2, 3, 5, 7]


# Return all supported image files from the input folder in sorted order.
def load_image_files(folder_path):
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    return sorted([f for f in os.listdir(folder_path) if f.lower().endswith(exts)])


# Load the YOLO object detection model and the SAM segmentation model.
def load_models(yolo_model_name="yolov8m.pt", sam_model_name="mobile_sam.pt"):
    yolo_model = YOLO(yolo_model_name)
    sam_model = SAM(sam_model_name)
    return yolo_model, sam_model


class FastMotionMaskEngine:
    # Initialize the motion estimation engine.
    # scale is used to process frames at a smaller resolution for faster computation.
    def __init__(self, scale=0.25):
        self.scale = scale

        # DIS optical flow is used to estimate pixel-level motion between two frames.
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

        # Variables used to cache the coordinate grid for the current image size.
        self.cached_shape = None
        self.gx = None
        self.gy = None
        self.coords = None

    # Create and cache a coordinate grid whenever the resized image dimensions change.
    def _update_grid_cache(self, h_s, w_s):
        if self.cached_shape != (h_s, w_s):
            self.gx, self.gy = np.meshgrid(np.arange(w_s), np.arange(h_s))

            # Store homogeneous coordinates for applying the homography.
            self.coords = np.stack(
                [self.gx, self.gy, np.ones_like(self.gx)],
                axis=-1
            ).reshape(-1, 3).T

            self.cached_shape = (h_s, w_s)

    # Estimate motion between two consecutive images.
    def compute_motion(self, img1, img2):
        h, w = img1.shape[:2]

        # Resize dimensions based on the configured processing scale.
        h_s, w_s = int(h * self.scale), int(w * self.scale)

        # Make sure the coordinate grid matches the current image size.
        self._update_grid_cache(h_s, w_s)

        # Convert both images to grayscale and resize them for faster processing.
        gray1_s = cv2.resize(
            cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
            (w_s, h_s)
        )
        gray2_s = cv2.resize(
            cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
            (w_s, h_s)
        )

        # Detect strong feature points in the first frame.
        p0 = cv2.goodFeaturesToTrack(
            gray1_s,
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=7
        )

        # If too few feature points are found, return an empty motion mask.
        if p0 is None or len(p0) < 8:
            return np.zeros((h_s, w_s), dtype=np.uint8)

        # Track the detected feature points into the second frame.
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            gray1_s,
            gray2_s,
            p0,
            None,
            winSize=(21, 21),
            maxLevel=3
        )

        # If optical flow tracking fails, return an empty motion mask.
        if p1 is None:
            return np.zeros((h_s, w_s), dtype=np.uint8)

        # Keep only feature points that were successfully tracked.
        good_p0 = p0[st == 1]
        good_p1 = p1[st == 1]

        # A minimum number of points is required to estimate the camera motion.
        if len(good_p0) < 8:
            return np.zeros((h_s, w_s), dtype=np.uint8)

        # Estimate the global image transformation using a homography.
        # RANSAC helps reject feature-point outliers.
        H, _ = cv2.findHomography(good_p0, good_p1, cv2.RANSAC, 4.0)

        # If a valid homography cannot be calculated, return an empty mask.
        if H is None:
            return np.zeros((h_s, w_s), dtype=np.uint8)

        # Calculate dense optical flow between the two frames.
        flow = self.dis.calc(gray1_s, gray2_s, None)

        # Apply the estimated global homography to every pixel coordinate.
        warped = H @ self.coords
        warped /= (warped[2, :] + 1e-6)

        # Calculate the expected motion caused by global camera movement.
        exp_dx = (
            warped[0, :] - self.gx.reshape(-1)
        ).reshape(h_s, w_s)

        exp_dy = (
            warped[1, :] - self.gy.reshape(-1)
        ).reshape(h_s, w_s)

        # Extract the actual optical flow displacement.
        act_dx = flow[:, :, 0]
        act_dy = flow[:, :, 1]

        # Residual motion represents movement that cannot be explained
        # by the estimated global camera motion.
        residual = np.sqrt(
            (act_dx - exp_dx)**2 +
            (act_dy - exp_dy)**2
        )

        # Convert the residual motion into a binary motion mask.
        motion_binary_s = (residual > 0.5).astype(np.uint8) * 255

        return motion_binary_s


def run_pipeline(input_folder, output_folder):
    # Create the output directory if it does not already exist.
    os.makedirs(output_folder, exist_ok=True)

    # Load and sort all image files from the input directory.
    files = load_image_files(input_folder)

    # At least two frames are required to calculate motion.
    if len(files) < 2:
        return

    # Load YOLO and SAM models.
    yolo_model, sam_model = load_models("yolov8m.pt", "mobile_sam.pt")

    # Create the motion estimation engine.
    engine = FastMotionMaskEngine(scale=0.25)

    # Use a thread pool so image writing can happen asynchronously.
    io_pool = ThreadPoolExecutor(max_workers=4)

    # Read the first frame to initialize the frame-to-frame processing.
    prev_img = cv2.imread(
        os.path.join(input_folder, files[0])
    )

    # Process every pair of consecutive frames.
    for i in range(1, len(files)):
        curr_file = files[i]
        curr_img = cv2.imread(
            os.path.join(input_folder, curr_file)
        )

        # 1. Compute motion between prev_img and curr_img.
        motion_binary_s = engine.compute_motion(prev_img, curr_img)

        # 2. Run YOLO object detection on the previous frame.
        results = yolo_model(
            prev_img,
            imgsz=640,
            conf=0.15,
            verbose=False
        )[0]

        moving_boxes = []

        # Check whether YOLO detected any objects.
        if results.boxes is not None and len(results.boxes) > 0:
            classes = results.boxes.cls.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy()

            # Examine every detected object.
            for cls, box in zip(classes, boxes):

                # Only consider classes that are defined as dynamic objects.
                if int(cls) in DYNAMIC_CLASSES:
                    x1, y1, x2, y2 = map(int, box)

                    # Convert bounding-box coordinates to the scaled
                    # resolution used by the motion estimation step.
                    x1_s, y1_s = int(x1 * engine.scale), int(y1 * engine.scale)
                    x2_s, y2_s = int(x2 * engine.scale), int(y2 * engine.scale)

                    # Ensure coordinates are within valid image bounds.
                    x1_s, y1_s = max(0, x1_s), max(0, y1_s)

                    # Extract the motion region inside the detected object box.
                    box_motion = motion_binary_s[
                        y1_s:y2_s,
                        x1_s:x2_s
                    ]

                    # Calculate the area of the bounding box.
                    box_area = (x2_s - x1_s) * (y2_s - y1_s)

                    if box_area > 0:
                        # Calculate the percentage of the object box that contains motion.
                        motion_ratio = np.sum(box_motion > 0) / box_area

                        # If more than 2% of the object region is moving,
                        # mark this object for SAM segmentation.
                        if motion_ratio > 0.02:
                            moving_boxes.append(box)

        # 3. Run SAM only on objects that have been identified as moving.
        h, w = prev_img.shape[:2]

        # Initialize an empty full-resolution mask.
        mask = np.zeros((h, w), dtype=np.uint8)

        if moving_boxes:

            # Use the detected moving bounding boxes as prompts for SAM.
            sam_results = sam_model(
                prev_img,
                bboxes=moving_boxes,
                verbose=False
            )[0]

            # Check whether SAM produced segmentation masks.
            if sam_results.masks is not None:
                masks = sam_results.masks.data.cpu().numpy()

                # Combine all generated masks into one binary mask.
                for msk in masks:
                    resized = cv2.resize(
                        msk,
                        (w, h),
                        interpolation=cv2.INTER_NEAREST
                    )

                    mask = np.maximum(
                        mask,
                        (resized > 0.5).astype(np.uint8) * 255
                    )

            # Close small gaps and holes in the generated mask.
            kernel_close = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (5, 5)
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel_close
            )

            # Slightly expand the mask to ensure the moving object
            # is completely covered.
            kernel_dilate = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (15, 15)
            )
            mask = cv2.dilate(
                mask,
                kernel_dilate,
                iterations=1
            )

        # Use the previous frame's filename as the output base name.
        base_name = os.path.splitext(files[i - 1])[0]

        # Create a copy of the previous frame for modification.
        cleaned_img = prev_img.copy()

        # Remove the detected moving regions by replacing them with black pixels.
        cleaned_img[mask > 0] = [0, 0, 0]

        # Define the output path for the cleaned frame.
        out_path = os.path.join(
            output_folder,
            f"{base_name}_cleaned.jpg"
        )

        # Write the image asynchronously using the thread pool.
        io_pool.submit(
            cv2.imwrite,
            out_path,
            cleaned_img
        )

        # Move to the next frame.
        prev_img = curr_img

    # Output the final frame without modification because there is no
    # subsequent frame available for motion comparison.
    base_name = os.path.splitext(files[-1])[0]

    out_path = os.path.join(
        output_folder,
        f"{base_name}_cleaned.jpg"
    )

    io_pool.submit(
        cv2.imwrite,
        out_path,
        prev_img
    )

    # Wait for all pending image-writing operations to finish.
    io_pool.shutdown(wait=True)


# Run the pipeline when this file is executed directly.
if __name__ == "__main__":
    run_pipeline(
        "./pipeline_input/keyframes",
        "./pipeline_output/masks"
    )