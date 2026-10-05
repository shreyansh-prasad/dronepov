# Single-Pass Drone Video to Georeferenced 3D Reconstruction

## Smart India Hackathon (SIH) Project

### Overview
This project converts a **single continuous drone/UAV video flight** into a **georeferenced and metrically useful 3D model** of the surveyed area.

Traditional photogrammetry often requires multiple drone passes and high image overlap. Our approach extracts the most useful information from one flight, reducing flight time and operational effort.

### Applications
- Rapid disaster assessment
- Infrastructure inspection
- Urban and terrain mapping
- Construction monitoring
- Archaeological documentation
- Road and building assessment
- Surveillance and situational awareness
- Digital-twin generation

## Problem
A detailed drone 3D model normally requires multiple passes, many overlapping images, careful flight planning and significant processing. In disasters, emergencies and inspections, repeated flights may not be possible.

The system therefore targets reconstruction from **one moving UAV video**, while handling motion blur, compression, changing illumination, shadows, GPS noise, occlusions, moving objects and limited visual overlap.

## Proposed Solution

```
Drone Video + GPS + Flight Metadata
                |
                v
        Video Processing
                |
                v
        Frame Quality Analysis
                |
                v
         Keyframe Selection
                |
                v
    Dynamic Object Detection
        & Object Tracking
                |
                v
        Dynamic Object Masking
                |
                v
       Camera Pose Estimation
                |
                v
        3D Reconstruction
                |
                v
      GPS/IMU Georeferencing
                |
                v
       Point Cloud / 3D Mesh
                |
                v
        Web-Based 3D Viewer
                |
                v
      Measurement & Export
```

## 1. Input Data

### Required
- Drone video
- GPS coordinates
- Flight metadata

### Optional
- IMU data
- Barometric altitude
- Camera intrinsic parameters
- RTK/PPK positioning data

Optional sensor information improves accuracy but is not mandatory for the basic pipeline.

## 2. Intelligent Frame Selection

A video contains many unnecessary frames such as blurry, dark, nearly identical or low-feature frames. The system evaluates each candidate using:

- Sharpness
- Brightness/exposure
- Number of visual features
- Similarity to selected frames
- Time interval
- Useful scene overlap

Only useful keyframes are passed to reconstruction.

## 3. Moving Object Removal

People, cars, animals and other moving objects can create incorrect 3D points. Object detection/segmentation and tracking identify dynamic objects and create masks over those regions instead of discarding frames entirely.

```
Original Frame
      |
      v
Object Detection
      |
      v
Dynamic Object Mask
      |
      v
Masked Frame
      |
      v
3D Reconstruction
```

Typical classes include people, cars, trucks, buses, motorcycles, bicycles and animals.

## 4. Camera Movement and Pose

The drone continuously moves while recording. Visual information between frames is used to estimate camera movement and relative camera poses. GPS and optional IMU information can constrain or improve these estimates.

The reconstruction is first created in a local coordinate system and can then be aligned to geographic coordinates.

## 5. 3D Reconstruction

The cleaned keyframes are processed using a Structure-from-Motion / Multi-View Stereo pipeline.

Outputs include:

- **Sparse point cloud** — important reconstructed 3D features
- **Dense point cloud** — detailed 3D representation
- **3D mesh** — connected surfaces for terrain and structures
- **Textured model** — visual appearance projected from drone imagery

The current prototype integrates **COLMAP** for camera pose estimation and sparse reconstruction, with dense reconstruction and meshing planned as later stages.

## 6. GPS and Georeferencing

GPS is used as geographic information rather than as a direct replacement for visual reconstruction.

```text
Local 3D Reconstruction
          +
GPS / IMU / RTK Information
          |
          v
Georeferenced 3D Model
```

The system can synchronize GPS with video timestamps, detect unrealistic jumps, handle missing samples, smooth noisy trajectories and use higher-quality RTK/PPK data when available.

Survey-grade accuracy should only be claimed when appropriate positioning data and independent validation support it.

## 7. Measurements

The georeferenced model can support:

- Distance between points
- Approximate building height
- Road width
- Elevation difference
- Approximate area
- Relative dimensions

Measurement confidence should be reported rather than assuming survey-grade accuracy.


## Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Video Processing | OpenCV |
| Numerical Processing | NumPy |
| Data Processing | Pandas |
| Object Detection / Segmentation | YOLO |
| Object Tracking | ByteTrack / BoT-SORT |
| Camera Pose / Sparse Reconstruction | COLMAP |
| 3D Processing | Open3D |
| GIS / Coordinates | PyProj / GeoPandas |
| Backend | FastAPI |
| Frontend | React |
| 3D Visualization | Three.js |
| Sensors | GPS / IMU / RTK / PPK |

## Drone Video Frame Selection Pipeline - Detailed Technical Documentation

### Overview of Frame Selection Strategy

The frame selection pipeline is a critical component that intelligently extracts keyframes from continuous drone video footage. This process reduces computational load while preserving essential visual information for high-quality 3D reconstruction.

### Frame Quality Assessment Metrics

The system evaluates frames based on multiple quality dimensions, including blur, exposure, texture, temporal consistency, and scene overlap. Sharpness is usually quantified by gradient- or Laplacian-based metrics.

#### 1. Variance of Laplacian

The Laplacian-based sharpness metric estimates the degree of edge content in a frame:

```text
S_lap(F_i) = Var(∇² F_i)
```

where:
- F_i is the i-th frame
- ∇² F_i is the Laplacian of the frame
- Var(·) is the variance across all pixel values

A higher value indicates stronger edge contrast and sharper image content.

#### 2. Tenengrad Method

The Tenengrad metric uses image gradients computed with Sobel operators:

```text
S_ten(F_i) = (1 / (M × N)) × Σx=1..M Σy=1..N sqrt(Gx(x, y)^2 + Gy(x, y)^2)
```

where:
- M and N are image dimensions
- Gx(x, y) and Gy(x, y) are the horizontal and vertical gradient responses at pixel (x, y)

This metric effectively measures high-frequency content, which is strongly correlated with sharpness.

#### 3. Brenner Gradient

The Brenner metric measures the difference between neighboring pixels along the vertical direction:

```text
S_bren(F_i) = Σx=1..M Σy=1..N-2 [F_i(x, y) - F_i(x, y + 2)]^2
```

where F_i(x, y) is the grayscale intensity at pixel position (x, y). Large differences indicate well-defined edges and improved sharpness.

#### 4. Combined Frame Quality Score

The selected keyframes are often ranked using a weighted combination of these metrics:

```text
Score(F_i) = w1 × S_lap(F_i) + w2 × S_ten(F_i) + w3 × S_bren(F_i) + w4 × S_texture(F_i) - λ × P_blur(F_i)
```

where:
- w1, w2, w3, w4 are weighting coefficients
- S_texture measures feature richness and local contrast
- P_blur is a penalty for motion blur or low-frequency content
- λ controls the strength of the blur penalty

### Frame Selection Algorithm

The intelligent frame selection follows a multi-stage approach:

1. Initial filtering to remove obviously poor frames based on sharpness and brightness
2. Feature extraction for candidate frames
3. Similarity analysis to reject near-duplicate frames
4. Temporal sampling to maintain useful spacing between selected frames
5. Overlap validation to ensure sufficient scene coverage for reconstruction

### Dynamic Object Detection and Masking

#### Object Detection Workflow
- YOLO-based detection during frame processing
- Multi-scale object detection for different spatial scales
- Confidence thresholding to minimize false positives
- Class-specific analysis for vehicles, people, and other moving objects

#### Masking Strategy
- Precise boundary delineation of detected objects
- Morphological operations for mask refinement
- Dilation and erosion routines for temporal consistency
- Weighted masking to preserve useful scene structure while removing dynamic content

#### Temporal Tracking
- Frame-to-frame tracking using ByteTrack or BoT-SORT
- Trajectory smoothing across consecutive frames
- Occlusion handling and tracked-object persistence
- Mask propagation across time for cleaner reconstruction

### GPS and IMU Integration

#### GPS Trajectory Processing
- Timestamp synchronization with video frames
- Noise filtering and outlier removal
- Trajectory smoothing for realistic motion estimation
- Uncertainty estimation for georeferencing

#### IMU Data Fusion
- Gyroscope-based rotation estimation
- Accelerometer processing for motion parameters
- Sensor synchronization and time alignment
- Drift correction using GPS anchors

#### Coordinate System Transformation
- Local coordinate system initialization
- GPS-to-local transformation
- Georeferencing through ground control points
- Projection and datum handling for geospatial output

### Output and Deliverables

#### Primary Outputs
- Keyframe sequence with metadata
- Feature maps and descriptors
- Camera pose estimates
- Sparse 3D point clouds
- Dynamic object masks

#### Quality Metrics
- Reconstruction confidence scores
- Feature matching statistics
- Geometric consistency measures
- Coverage area assessment

#### Export Formats
- Keyframe images (PNG/JPEG)
- Feature files
- Point cloud data (PLY/LAS)
- Camera parameter matrices
- Transformation matrices
- Metadata JSON files

### Performance Optimization

#### Computational Efficiency
- GPU acceleration for feature extraction
- Parallel frame processing
- Adaptive resolution for previews
- Memory-efficient data structures

#### Scalability
- Streaming processing for large videos
- Tile-based processing for large areas
- Incremental reconstruction support
- Distributed processing capabilities

### Quality Assurance and Validation

#### Reconstruction Validation
- Epipolar geometry consistency checks
- Triangulation angle assessment
- Reprojection error analysis
- Photometric consistency validation

#### Confidence Metrics
- Per-feature confidence scores
- Per-point confidence levels
- Coverage completeness assessment
- Geometric stability indices

### Best Practices and Guidelines

#### Optimal Flight Parameters
- Flight altitude: 30–150 meters for balanced resolution and coverage
- Speed: 5–15 m/s to preserve image stability
- Overlap: minimum 60% forward overlap for reconstruction robustness
- Coverage planning: maximize feature-rich, textured areas

#### Data Acquisition Tips
- Fly during optimal lighting conditions
- Avoid extreme weather and poor visibility
- Maintain consistent altitude when possible
- Plan multiple passes for complex terrain if needed

#### Processing Recommendations
- Start with conservative quality thresholds
- Validate intermediate outputs frequently
- Use reference imagery when available
- Document processing parameters for reproducibility

## Current Prototype

### Phase 1 — Intelligent Frame Selection
Implemented:
- Video reading and frame sampling
- Sharpness and brightness analysis
- Feature detection
- Similar-frame filtering
- Keyframe selection
- Processing statistics


### Phase 2 — Dynamic Object Detection
Implemented:
- Object detection/segmentation
- Dynamic-object identification
- Object tracking
- Dynamic-object masking
- Detection previews and statistics


### Phase 3 — Sparse 3D Reconstruction
Current work includes:
- COLMAP integration
- Feature extraction
- Image matching
- Camera registration
- Sparse reconstruction
- Point-cloud export

The reconstruction stage is being improved for difficult single-pass footage and low-overlap situations.


### Phase 4 & 5 — Image-Only 3D Reconstruction Pipeline (Dense MVS, Poisson Meshing, UV Texturing & Export)

This module provides a complete, end-to-end 3D reconstruction pipeline that transforms ordered video keyframes directly into textured, self-contained 3D models (`.glb` / `.obj`), dense point clouds, and detailed reports.

#### Key Highlights & Workflow:
- **Input**: Ordered drone video keyframes (`.png` / `.jpg`).
- **Image-Only Mode (No GPS/Telemetry Required)**: Camera poses and trajectory are estimated entirely from imagery using sequential Structure-from-Motion (COLMAP). Flight telemetry, GPS coordinates, and IMU data are optional enhancements.
- **Dense MVS Reconstruction**: OpenMVS / PatchMatch pipeline generating dense 3D point representations.
- **Surface Meshing**: Screened Poisson Surface Reconstruction with boundary density trimming and manifold hole filling.
- **Photographic UV Texture Mapping**: UV unwrapping with `xatlas` and projective camera ray texture baking into a 2048×2048 texture atlas embedded directly in a self-contained GLB.
- **Quality & Confidence Tagging**: Per-vertex reconstruction confidence computed across multi-view ray density and camera proximity.
- **Deliverables Export**:
  - `mesh_full.glb`: Self-contained 3D model with embedded photographic UV texture atlas.
  - `mesh_decimated.glb`: Optimized lightweight mesh for web/real-time viewers.
  - `mesh_confidence.glb`: Quality-tagged 3D mesh with per-vertex reconstruction confidence.
  - `pointcloud_dense.ply` & `pointcloud_dense.las`: Cleaned dense point cloud.
  - `summary.html` & `report.md`: Interactive visual summary and accuracy audit report.

#### Quickstart: Running Image-Only Reconstruction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run reconstruction on extracted frames
python run_pipeline.py --input_dir sample_data/frames --output_dir output_image_only
```



## Key Innovation

The innovation is combining several steps specifically for the constraints of a **single continuous drone flight**:

1. **Smart frame selection** — process useful frames instead of the complete video.
2. **Dynamic-object removal** — reduce unwanted reconstruction from moving objects.
3. **Sensor-assisted georeferencing** — use GPS and optional IMU/RTK/PPK information.
4. **Automated reconstruction** — reduce manual photogrammetry work.
5. **Measurement-ready output** — support basic geographic and dimensional analysis.

## Advantages

- Requires only one continuous drone pass
- Reduces unnecessary frames
- Handles moving objects
- Reduces manual preprocessing
- Can use standard drone GPS
- Can incorporate better positioning sensors
- Produces a 3D representation for visualization and analysis
- Modular design allows individual components to be improved independently

## Limitations

Single-pass reconstruction has fundamental limitations. Results depend on camera quality, video resolution, motion blur, scene texture, camera movement, visual overlap, lighting, occlusion, GPS quality and other factors.

Very uniform surfaces, severe blur, insufficient overlap or areas never viewed by the camera may not reconstruct reliably.

The system is therefore aimed at **rapid mapping and situational awareness**. Survey-grade applications require suitable high-accuracy positioning and independent validation.

## Future Improvements

- More robust single-pass image matching
- Improved camera trajectory estimation
- Dense Multi-View Stereo
- Automatic mesh generation
- Texture generation
- Better GPS trajectory filtering
- IMU fusion
- RTK/PPK integration
- Improved dynamic-object handling
- Terrain/building classification
- Automatic building and road extraction
- Accuracy/confidence reporting
- GPU acceleration
- Large-area processing
- Cloud deployment
- Interactive web-based 3D visualization



## Project Goal

The long-term goal is to turn **one drone flight into actionable 3D geographic information**, reducing repeated flights while providing a fast and automated reconstruction workflow.
