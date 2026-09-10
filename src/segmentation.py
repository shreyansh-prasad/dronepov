"""
Stage 2 — Semantic Segmentation for Aerial/Drone Imagery

Hybrid approach:
  1. HSV + spatial colour-space analysis for aerial-specific classes
     (sky, vegetation, road, building, bare ground, water)
  2. COCO DeepLab for dynamic object detection (cars, people, animals)
  3. Merge both into a unified multi-class mask

UAVid-equivalent classes we produce:
  0 = Background/Clutter
  1 = Building
  2 = Road
  3 = Tree / High Vegetation
  4 = Low Vegetation / Grass
  5 = Static Car (from COCO)
  6 = Moving Car (from COCO, treated same for now)
  7 = Human (from COCO)
  8 = Sky
  9 = Water
  10 = Bare Ground / Dirt
"""

import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
import numpy as np
import cv2
import logging
from PIL import Image

logger = logging.getLogger("Stage2_Segmentation")

# ─── Colour palette for 11 aerial classes ───────────────────────────
#   Index   Class              Colour (BGR for OpenCV)
AERIAL_PALETTE = np.array([
    [80,  80,  80],    # 0  Background/Clutter        — dark grey
    [0,   0,   180],   # 1  Building                   — red
    [128, 128, 128],   # 2  Road                       — medium grey
    [0,   128, 0],     # 3  Tree / High Vegetation     — dark green
    [0,   255, 128],   # 4  Low Vegetation / Grass     — light green
    [255, 0,   0],     # 5  Static Car                 — blue
    [255, 0,   255],   # 6  Moving Car                 — magenta
    [0,   255, 255],   # 7  Human                      — yellow
    [255, 200, 150],   # 8  Sky                        — light blue
    [200, 100, 0],     # 9  Water                      — dark blue
    [60,  120, 180],   # 10 Bare Ground / Dirt          — brown
], dtype=np.uint8)

AERIAL_CLASS_NAMES = [
    "background", "building", "road", "tree", "low_veg",
    "static_car", "moving_car", "human", "sky", "water", "bare_ground"
]

# Dynamic classes to EXCLUDE from 3D reconstruction
DYNAMIC_CLASSES = {6, 7}  # moving_car, human
# Classes to mask OUT entirely from reconstruction
EXCLUDE_FROM_RECON = {6, 7, 8}  # moving_car, human, sky


def mask_to_color_bgr(mask_2d):
    """Convert a 2D class-ID mask into a BGR colour image using aerial palette."""
    h, w = mask_2d.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(min(len(AERIAL_PALETTE), int(mask_2d.max()) + 1)):
        colour[mask_2d == cls_id] = AERIAL_PALETTE[cls_id]
    return colour  # already BGR


class SegmentationAgent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading segmentation model on {self.device}")

        # COCO model — only used for dynamic object detection (cars, people)
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        self.coco_model = deeplabv3_mobilenet_v3_large(weights=weights).to(self.device).eval()
        self.coco_transforms = weights.transforms()

        # COCO class → our aerial class mapping
        # person=1→7, car=3→5, motorcycle=4→6, bus=6→6, truck=8→5, bicycle=2→6
        self.coco_to_aerial = {
            1: 7,   # person → human
            2: 6,   # bicycle → moving_car (dynamic)
            3: 5,   # car → static_car
            4: 6,   # motorcycle → moving_car (dynamic)
            6: 6,   # bus → moving_car (dynamic)
            7: 6,   # train → moving_car (dynamic)
            8: 5,   # truck → static_car
        }

    def _segment_aerial_hsv(self, bgr_frame):
        """
        Colour-space + spatial segmentation tuned for aerial/drone imagery.
        Returns a 2D class mask using our 11 aerial classes.
        """
        # First principle: Apply edge-preserving smoothing to reduce pixel-level noise 
        # and create more contiguous, homogeneous semantic regions.
        bgr_smooth = cv2.bilateralFilter(bgr_frame, d=7, sigmaColor=50, sigmaSpace=50)

        h, w = bgr_smooth.shape[:2]
        hsv = cv2.cvtColor(bgr_smooth, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr_smooth, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(bgr_smooth, cv2.COLOR_BGR2GRAY)

        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # Start with everything as background
        mask = np.zeros((h, w), dtype=np.uint8)

        # ──── Sky Detection ────────────────────────────────────────────
        # Sky: high brightness, low saturation, or blue hue.
        # First principle: In purely top-down aerial shots, sky is rare.
        # We should not use a vertical gradient prior (top 40%) because the top of the image is still ground.
        # Require extremely bright values (clouds) or clear blue to be sky.
        sky_bright = (V > 220) & (S < 40)
        sky_blue = (H >= 90) & (H <= 135) & (S > 50) & (V > 150)

        sky_mask = sky_bright | sky_blue
        mask[sky_mask] = 8

        # ──── Vegetation Detection ─────────────────────────────────────
        # Green hue in HSV with reasonable saturation
        green_hue = (H >= 25) & (H <= 90)
        green_sat = S > 20
        green_mask = green_hue & green_sat & (~sky_mask)

        # Distinguish tree vs low vegetation by texture (high texture = tree canopy)
        # Use Laplacian variance in local patches
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        # Compute local variance using box filter
        lap_sq = lap * lap
        kernel_size = 15
        local_var = cv2.blur(lap_sq, (kernel_size, kernel_size))

        high_texture_veg = green_mask & (local_var > 200)
        low_texture_veg = green_mask & (local_var <= 200)

        # Dark green with high texture → tree
        dark_green = green_mask & (V < 180) & (S > 40)
        trees = green_mask & (high_texture_veg | dark_green)
        grass = green_mask & ~trees

        mask[trees] = 3   # tree
        mask[grass] = 4   # low vegetation / grass

        # ──── Road Detection ───────────────────────────────────────────
        # Roads: low saturation, medium brightness, smooth texture
        # First principle: Asphalt shadows are dark, smooth, and can have a blue tint (H:90-140) 
        # due to sky reflection and AWB. They shouldn't be classified as water.
        # Allow V up to 255 so that pure white road markings (V=255) are classified as road, not buildings.
        road_grey = (S < 70) & (V > 30)
        road_shadow = (S >= 70) & (S < 130) & (H >= 90) & (H <= 140) & (V > 20) & (V < 120)
        
        road_base = road_grey | road_shadow
        road_smooth = local_var < 100
        road_not_sky = ~sky_mask
        road_mask = road_base & road_smooth & road_not_sky & ~green_mask

        # Morphological cleanup for roads (roads are large connected areas)
        road_uint8 = road_mask.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        road_uint8 = cv2.morphologyEx(road_uint8, cv2.MORPH_CLOSE, kernel)
        road_uint8 = cv2.morphologyEx(road_uint8, cv2.MORPH_OPEN, kernel)
        road_mask = road_uint8 > 0

        mask[road_mask] = 2

        # ──── Building Detection ───────────────────────────────────────
        # Buildings from above: structured edges, distinct colour patches,
        # NOT green, NOT sky, NOT road-smooth-grey
        # Detect via edge density + colour non-uniformity
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.blur(edges.astype(np.float32), (21, 21))

        # Buildings: medium-high edge density, not vegetation, not sky, not road
        building_candidates = (
            (edge_density > 15) &
            ~green_mask &
            ~sky_mask &
            ~road_mask &
            (S > 10) &  # some colour, not pure grey (roads)
            (V > 30)    # not too dark
        )

        # Rooftop colours: reddish-brown, grey-blue, white-ish
        red_brown = (H < 25) | (H > 160)  # reddish hues (wraps around)
        warm_tone = (A > 128) & (B > 128)  # warm in LAB
        rooftop_colour = red_brown | warm_tone | ((S < 60) & (V > 80))

        building_mask = building_candidates & rooftop_colour

        # Morphological cleanup
        bld_uint8 = building_mask.astype(np.uint8) * 255
        # First principle: Edge detection only finds the OUTLINE of the building.
        # We must use a massively large closing kernel to fuse the edges together and fill the hollow roof interior!
        kernel_bld = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
        bld_uint8 = cv2.morphologyEx(bld_uint8, cv2.MORPH_CLOSE, kernel_bld)
        building_mask = bld_uint8 > 0

        mask[building_mask] = 1

        # ──── Water Detection ──────────────────────────────────────────
        # Dark blue/teal, very low texture, not sky, not road.
        # Must be highly saturated to distinguish from dark asphalt shadows.
        water = (
            (H >= 85) & (H <= 140) &
            (S > 130) &
            (V < 180) &
            (local_var < 50) &
            ~sky_mask & ~road_mask & ~building_mask
        )
        
        # Water is usually a large contiguous body, filter out tiny noise patches
        water_uint8 = water.astype(np.uint8) * 255
        kernel_water = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        water_uint8 = cv2.morphologyEx(water_uint8, cv2.MORPH_OPEN, kernel_water)
        water = water_uint8 > 0
        
        mask[water] = 9

        # ──── Bare Ground / Dirt ───────────────────────────────────────
        # Brown/tan: low saturation warmish hue, not already classified
        bare = (
            ((H >= 10) & (H <= 30)) &
            (S > 30) & (S < 120) &
            (V > 60) & (V < 200) &
            (mask == 0)  # only unclassified pixels
        )
        mask[bare] = 10

        return mask

    def _detect_dynamic_objects_coco(self, bgr_frame):
        """
        Use COCO DeepLab specifically for dynamic object detection.
        Applies a strict confidence threshold and morphological cleanup to prevent 
        hallucinating buses/trains on buildings and shadows from aerial perspectives.
        """
        rgb = bgr_frame[:, :, ::-1]
        pil_img = Image.fromarray(rgb)
        input_tensor = self.coco_transforms(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_logits = self.coco_model(input_tensor)['out'][0]

        # Calculate softmax probabilities to filter low-confidence hallucinations
        probs = torch.nn.functional.softmax(output_logits, dim=0)
        max_probs, coco_mask = torch.max(probs, dim=0)

        coco_mask = coco_mask.byte().cpu().numpy()
        max_probs = max_probs.cpu().numpy()

        # First principle: Pre-trained COCO models struggle with top-down aerial views of cars.
        # Even at 45% it was missing obvious cars. We lower to 20% to force detections, 
        # and rely heavily on the morphological opening below to destroy the noisy false positives.
        CONFIDENCE_THRESHOLD = 0.20
        coco_mask[max_probs < CONFIDENCE_THRESHOLD] = 0

        # Map COCO detections to our aerial classes
        h, w = bgr_frame.shape[:2]
        # Resize COCO mask to match frame if needed
        if coco_mask.shape != (h, w):
            coco_mask = cv2.resize(coco_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        aerial_dynamic = np.zeros((h, w), dtype=np.uint8)
        kernel_dyn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        for coco_cls, aerial_cls in self.coco_to_aerial.items():
            mask_cls = (coco_mask == coco_cls).astype(np.uint8)
            # Remove impossibly tiny dynamic object fragments
            mask_cls = cv2.morphologyEx(mask_cls, cv2.MORPH_OPEN, kernel_dyn)
            
            aerial_dynamic[mask_cls > 0] = aerial_cls

        return aerial_dynamic

    def segment(self, full_frame):
        """
        Hybrid segmentation: HSV aerial analysis + COCO dynamic object detection.

        Returns:
            mask: 2D array of aerial class IDs (11 classes)
            dynamic_mask: 2D boolean array (True where dynamic objects are)
            mission_value: float score based on structural content richness
        """
        # Step 1: Aerial colour-space segmentation
        aerial_mask = self._segment_aerial_hsv(full_frame)

        # Step 2: COCO dynamic object overlay (cars, people override aerial mask)
        dynamic_overlay = self._detect_dynamic_objects_coco(full_frame)
        # Dynamic objects override whatever the HSV classifier said
        has_dynamic = dynamic_overlay > 0
        aerial_mask[has_dynamic] = dynamic_overlay[has_dynamic]

        # Step 3: Compute outputs
        dynamic_mask = np.isin(aerial_mask, list(DYNAMIC_CLASSES))

        # Mission value: proportion of structurally interesting pixels
        # Buildings, roads, and cars are high value; sky and bare ground are low
        high_value_pixels = np.isin(aerial_mask, [1, 2, 3, 5])  # building, road, tree, static_car
        medium_value_pixels = np.isin(aerial_mask, [4, 10])      # low veg, bare ground
        mission_value = (
            float(np.sum(high_value_pixels)) / aerial_mask.size * 1.0 +
            float(np.sum(medium_value_pixels)) / aerial_mask.size * 0.3
        )

        return aerial_mask, dynamic_mask, mission_value

    def save_mask_overlay(self, full_frame, mask, save_path, alpha=0.50):
        """
        Save a colour overlay: original frame blended with coloured semantic mask.
        Also draws a small legend in the corner.
        """
        colour_mask_bgr = mask_to_color_bgr(mask)

        # Resize if needed
        if colour_mask_bgr.shape[:2] != full_frame.shape[:2]:
            colour_mask_bgr = cv2.resize(
                colour_mask_bgr,
                (full_frame.shape[1], full_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        overlay = cv2.addWeighted(full_frame, 1 - alpha, colour_mask_bgr, alpha, 0)

        # Draw compact legend in top-right corner
        present_classes = np.unique(mask)
        legend_h = 18 * len(present_classes) + 10
        legend_w = 160
        lx = full_frame.shape[1] - legend_w - 10
        ly = 10

        # Semi-transparent background
        sub = overlay[ly:ly + legend_h, lx:lx + legend_w].copy()
        cv2.rectangle(overlay, (lx, ly), (lx + legend_w, ly + legend_h), (0, 0, 0), -1)
        overlay[ly:ly + legend_h, lx:lx + legend_w] = cv2.addWeighted(
            sub, 0.3, overlay[ly:ly + legend_h, lx:lx + legend_w], 0.7, 0
        )

        for i, cls_id in enumerate(present_classes):
            if cls_id >= len(AERIAL_CLASS_NAMES):
                continue
            cy = ly + 15 + i * 18
            colour = AERIAL_PALETTE[cls_id].tolist()
            cv2.rectangle(overlay, (lx + 5, cy - 8), (lx + 17, cy + 4), colour, -1)
            cv2.putText(overlay, AERIAL_CLASS_NAMES[cls_id], (lx + 22, cy + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(save_path, overlay)
