# Single-Pass Drone Video to Georeferenced 3D Reconstruction

## Smart India Hackathon (SIH) Project

### Overview

This project converts a *single continuous drone/UAV video flight* into a *georeferenced and metrically useful 3D model* of the surveyed area.


Traditional photogrammetry often requires multiple drone passes and high image overlap. Our approach extracts the most useful information from one flight, reducing flight time and operational effort while producing a useful 3D representation.


### Applications


•Rapid disaster assessment

•Infrastructure inspection

•Urban and terrain mapping

•Construction monitoring

•Archaeological documentation

•Road and building assessment

•Surveillance and situational awareness

•Digital-twin generation


## Problem

A detailed drone 3D model normally requires multiple passes, many overlapping images, careful flight planning and significant processing. In disasters, emergencies and inspections, repeated flights may not be practical.


The system therefore targets reconstruction from *one moving UAV video*, while handling motion blur, compression, changing illumination, shadows, GPS noise, occlusions, moving objects and limited viewing angles.


## Proposed Solution

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

## 1. Input Data

## Required


•Drone video

•GPS coordinates

•Flight metadata


## Optional


•IMU data

•Barometric altitude

•Camera intrinsic para meters

•RTK/PPK positioning data


Optional sensor information improves accuracy but is not mandatory for the basic pipeline.


## 2. Intelligent Frame Selection

A video contains many unnecessary frames such as blurry, dark, nearly identical or low-feature frames. The system evaluates each candidate using:



•Sharpness

•Brightness/exposure

•Number of visual features

•Similarity to selected frames

•Time interval

•Useful scene overlap


Only useful keyframes are passed to reconstruction.


## 2.1 Mathematical Formulation of Intelligent Frame Selection

The intelligent frame-selection stage is formulated as a two-step optimization pipeline:


Raw Drone Video
      |
      v
Adaptive Quality Gate
      |
      v
Gated Candidate Frames
      |
      v
Submodular / Lazy-Greedy Selection
      |
      v
Final Keyframes + Masks + Confidence Metadata

## 2.1.1 Dynamic Blur Calibration

Instead of relying on a fixed blur threshold, the system calibrates the threshold from a representative sample of the input video:


$$
T_{\mathrm{blur}}

\max
\left(
10.0,;
0.5
\times
\operatorname{median}{t\in S{\mathrm{sample}}}
\left[
\operatorname{Var}
\left(
\nabla^2 I_t
\right)
\right]
\right)
$$


Where:


Symbol	Meaning
$T_{\mathrm{blur}}$	Dynamically calculated blur threshold
$I_t$	Frame at time $t$
$\nabla^2 I_t$	Laplacian of the image
$\operatorname{Var}(\nabla^2 I_t)$	Laplacian variance used as a sharpness measure
$S_{\mathrm{sample}}$	Sampled frames used for threshold calibration

A lower Laplacian variance indicates a less sharp frame.


## 2.1.2 Adaptive Acceptance / Salvage Rule

A low-quality frame is not automatically discarded. If a blurry or clipped frame provides coverage of an otherwise underrepresented region, it can be retained with a Low-Confidence flag.


$$
\operatorname{Accept}(I_t)

\begin{cases}
\text{True (Low-Confidence)}, &
\begin{aligned}
&\text{if }(\operatorname{IsBlurry}(I_t)
\lor \operatorname{IsClipped}(I_t))\
&\quad\land\operatorname{IsUnderCovered}(\operatorname{FOV}(I_t))
\end{aligned}
\[12pt]
\text{False (Quality Reject)}, &
\begin{aligned}
&\text{if }(\operatorname{IsBlurry}(I_t)
\lor \operatorname{IsClipped}(I_t))\
&\quad\land\neg\operatorname{IsUnderCovered}(\operatorname{FOV}(I_t))
\end{aligned}
\[12pt]
\text{False (Duplicate)}, &
\text{if }\operatorname{Overlap}(I_t,I_{\mathrm{anchor}})>0.85
\[8pt]
\text{True (Accepted)}, & \text{otherwise}
\end{cases}
$$


This allows the system to preserve frames that may be imperfect visually but are important because they cover otherwise missing areas.


## 2.1.3 Candidate Frame Set

After quality gating, the surviving frames form the candidate set:


$$
\mathcal{F}_{\mathrm{gated}}

{f_1,f_2,\ldots,f_n}
$$


Only these frames are passed to the second-stage keyframe-selection algorithm.


##2.1.4 Frame Usefulness Score

For a candidate frame $f_i$, its usefulness depends on the additional spatial coverage it provides, its mission value and its processing cost:


$$
U(f_i\mid\mathcal{S})

\underbrace{\Delta\mathcal{C}(f_i\mid\mathcal{S})}{\text{Marginal Area Gain}}
+
\underbrace{M(f_i)}{\text{Mission Value}}

\underbrace{\lambda_{\mathrm{cost}}}_{\text{Processing Cost}}
$$


Here, $\mathcal{S}$ represents the set of frames already selected.


## 2.1.5 Marginal Area Coverage Gain

The additional coverage contributed by a candidate frame is:


$$
\Delta\mathcal{C}(f_i\mid\mathcal{S})

\sum_{c\in\Omega}
\min
\left(
1,;
\mathcal{C}{\mathcal{S}}(c)
+
\mathbf{1}{f_i}(c)
\right)

\sum_{c\in\Omega}
\min
\left(
1,;
\mathcal{C}_{\mathcal{S}}(c)
\right)
$$


Where:


Symbol	Meaning
$\Omega$	Set of spatial cells or regions
$\mathcal{C}_{\mathcal{S}}(c)$	Existing coverage of cell $c$ by selected frames
$\mathbf{1}_{f_i}(c)$	Indicates whether frame $f_i$ covers cell $c$
$\Delta\mathcal{C}$	New coverage obtained by adding $f_i$

The formulation encourages spatial diversity: frames that introduce new scene regions receive a higher marginal gain, while frames that largely duplicate existing coverage receive a smaller gain.


## 2.1.6 Mission Value

The mission-specific value of each frame is defined as:


$$
M(f_i)

w_1\cdot
\operatorname{ClassDensity}_{\mathrm{UAVid}}(f_i)
+
w_2\cdot
\left(1-\operatorname{SkyRatio}(f_i)\right)
+
w_3\cdot
\left(1-\operatorname{DynamicRatio}(f_i)\right)
$$


Where:


Component	Purpose
$\operatorname{ClassDensity}_{\mathrm{UAVid}}$	Rewards frames containing relevant detected object classes
$1-\operatorname{SkyRatio}$	Rewards frames containing useful ground and scene content rather than excessive sky
$1-\operatorname{DynamicRatio}$	Rewards relatively static scene content
$w_1,w_2,w_3$	Tunable mission-specific weights

This makes the selection process aware of both geometric coverage and mission relevance.


## 2.1.7 Lazy-Greedy Selection

The final keyframe-selection problem is expressed as:


$$
\max_{\substack{
\mathcal{S}\subseteq\mathcal{F}{\mathrm{gated}}\
|\mathcal{S}|\le K
}}
;
\sum{i=1}^{|\mathcal{S}|}
U
\left(
f_i\mid\mathcal{S}_{i-1}
\right)
$$


where:


$$
\mathcal{S}_{i-1}

{f_1,\ldots,f_{i-1}}
$$


At each iteration, the algorithm prioritizes candidates with the highest estimated marginal usefulness and continues until the frame budget $K$ is reached or no remaining candidate provides sufficient additional value.


## 2.1.8 Complete Mathematical Pipeline

The complete frame-selection process can be summarized as:


$$
\boxed{
\text{Raw Video}
\xrightarrow{\text{Adaptive Quality Gate}}
\mathcal{F}_{\mathrm{gated}}
\xrightarrow{\text{Lazy-Greedy Selection}}
\mathcal{S}^{*}
}
$$


where:


$$
\mathcal{F}_{\mathrm{gated}}

\left{
f_i:
\operatorname{Accept}(f_i)=\mathrm{True}
\right}
$$


and:


$$
\mathcal{S}^{*}

\arg\max_{\substack{
\mathcal{S}\subseteq\mathcal{F}{\mathrm{gated}}\
|\mathcal{S}|\le K
}}
;
\sum{i=1}^{|\mathcal{S}|}
\left[
\Delta\mathcal{C}(f_i\mid\mathcal{S}_{i-1})
+
M(f_i)

\lambda_{\mathrm{cost}}
\right]
$$


The resulting set contains the selected keyframes together with their associated masks, coverage information and quality/confidence metadata.


## 3. Moving Object Removal

People, cars, animals and other moving objects can create incorrect 3D points. Object detection/segmentation and tracking identify dynamic objects and create masks over those regions instead of discarding the whole frame.


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

Typical classes include people, cars, trucks, buses, motorcycles, bicycles and animals.


## 4. Camera Movement and Pose

The drone continuously moves while recording. Visual information between frames is used to estimate camera movement and relative camera poses. GPS and optional IMU information can constrain or improve the trajectory.


The reconstruction is first created in a local coordinate system and can then be aligned to geographic coordinates.


## 5. 3D Reconstruction

The cleaned keyframes are processed using a Structure-from-Motion / Multi-View Stereo pipeline.


Outputs include:



•*Sparse point cloud* — important reconstructed 3D features

•*Dense point cloud* — detailed 3D representation

•*3D mesh* — connected surfaces for terrain and structures

•*Textured model* — visual appearance projected from drone imagery


The current prototype integrates COLMAP for camera pose estimation and sparse reconstruction, with dense reconstruction and meshing planned as later stages.


## 6. GPS and Georeferencing

GPS is used as geographic information rather than as a direct replacement for visual reconstruction.


Local 3D Reconstruction
          +
GPS / IMU / RTK Information
          |
          v
Georeferenced 3D Model

The system can synchronize GPS with video timestamps, detect unrealistic jumps, handle missing samples, smooth noisy trajectories and use higher-quality RTK/PPK data when available.


Survey-grade accuracy should only be claimed when appropriate positioning data and independent validation support it.


## 7. Measurements

The georeferenced model can support:



•Distance between points

•Approximate building height

•Road width

•Elevation difference

•Approximate area

•Relative dimensions


Measurement confidence should be reported rather than assuming survey-grade accuracy.


## 8. Web-Based 3D Viewer

A web interface can allow users to:



•Rotate, zoom and pan

•View point clouds and meshes

•View textures

•Select locations

•Measure distances

•Inspect geographic information

•Export reconstruction results


Possible architecture:


React
  |
  v
Three.js / WebGL
  |
  v
3D Point Cloud / Mesh

Technology Stack

Component	Technology
Programming	Python
Video Processing	OpenCV
Numerical Processing	NumPy
Data Processing	Pandas
Object Detection / Segmentation	YOLO
Object Tracking	ByteTrack / BoT-SORT
Camera Pose / Sparse Reconstruction	COLMAP
3D Processing	Open3D
GIS / Coordinates	PyProj / GeoPandas
Backend	FastAPI
Frontend	React
3D Visualization	Three.js
Sensors	GPS / IMU / RTK / PPK

Current Prototype

### Phase 1 — Intelligent Frame Selection

Implemented:



•Video reading and frame sampling

•Sharpness and brightness analysis

•Feature detection

•Similar-frame filtering

•Keyframe selection

•Processing statistics


### Phase 2 — Dynamic Object Detection

Implemented:



•Object detection/segmentation

•Dynamic-object identification

•Object tracking

•Dynamic-object masking

•Detection previews and statistics


### Phase 3 — Sparse 3D Reconstruction

Current work includes:



•COLMAP integration

•Feature extraction

•Image matching

•Camera registration

•Sparse reconstruction

•Point-cloud export


The reconstruction stage is being improved for difficult single-pass footage and low-overlap situations.



## Key Innovation

The innovation is combining several steps specifically for the constraints of a single continuous drone flight:



•*Smart frame selection* — process useful frames instead of the complete video.

•*Dynamic-object removal* — reduce unwanted reconstruction from moving objects.

•*Sensor-assisted georeferencing* — use GPS and optional IMU/RTK/PPK information.

•*Automated reconstruction* — reduce manual photogrammetry work.

•*Measurement-ready output* — support basic geographic and dimensional analysis.


## Advantages


•Requires only one continuous drone pass

•Reduces unnecessary frames

•Handles moving objects

•Reduces manual preprocessing

•Can use standard drone GPS

•Can incorporate better positioning sensors

•Produces a 3D representation for visualization and analysis

•Modular design allows individual components to be improved independently


## Limitations

•Single-pass reconstruction has fundamental limitations. Results depend on camera quality, video resolution, motion blur, scene texture, camera movement, visual overlap, lighting, occlusion, GPS quality and viewing angles.


•Very uniform surfaces, severe blur, insufficient overlap or areas never viewed by the camera may not reconstruct reliably.


•The system is therefore aimed at rapid mapping and situational awareness. Survey-grade applications require suitable high-accuracy positioning and independent validation.


## Future Improvements


•More robust single-pass image matching

•Improved camera trajectory estimation

•Dense Multi-View Stereo

•Automatic mesh generation

•Texture generation

•Better GPS trajectory filtering

•IMU fusion

•RTK/PPK integration

•Improved dynamic-object handling

•Terrain/building classification

•Automatic building and road extraction

•Accuracy/confidence reporting

•GPU acceleration

•Large-area processing

•Cloud deployment

•Interactive web-based 3D visualization


## Project Goal

The long-term goal is to turn one drone flight into actionable 3D geographic information, reducing repeated flights while providing a fast and automated reconstruction workflow.

