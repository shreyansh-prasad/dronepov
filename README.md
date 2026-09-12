# Single-Pass Drone Video to Georeferenced 3D Reconstruction

## Smart India Hackathon (SIH) Project

### Overview
This project converts a **single continuous drone/UAV video flight** into a **georeferenced and metrically useful 3D model** of the surveyed area.

Traditional photogrammetry often requires multiple drone passes and high image overlap. Our approach extracts the most useful information from one flight, reducing flight time and operational effort while producing a useful 3D representation.

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
A detailed drone 3D model normally requires multiple passes, many overlapping images, careful flight planning and significant processing. In disasters, emergencies and inspections, repeated flights may not be practical.

The system therefore targets reconstruction from **one moving UAV video**, while handling motion blur, compression, changing illumination, shadows, GPS noise, occlusions, moving objects and limited viewing angles.

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

People, cars, animals and other moving objects can create incorrect 3D points. Object detection/segmentation and tracking identify dynamic objects and create masks over those regions instead of discarding the whole frame.

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

The drone continuously moves while recording. Visual information between frames is used to estimate camera movement and relative camera poses. GPS and optional IMU information can constrain or improve the trajectory.

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

Single-pass reconstruction has fundamental limitations. Results depend on camera quality, video resolution, motion blur, scene texture, camera movement, visual overlap, lighting, occlusion, GPS quality and viewing angles.

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

