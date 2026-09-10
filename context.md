# Context — SIH25158 Frame Selection & Ranking Module

## Problem
SIH25158: generate a georeferenced, metrically accurate 3D model from a single-pass
drone video. Multiple sub-teams split the pipeline; this module owns one subtask:
turn a raw drone video into a ranked, gated subset of frames (+ per-frame metadata)
that another team's SfM/MVS reconstruction pipeline consumes.

## Constraints (fixed — not open for redesign)
- Runs offline, after landing. Not real-time.
- Local GPU laptop. No cloud, no multi-GPU.
- No GPS/IMU in the current prototype — video-only is the default input path, not
  an edge case.
- Reconstruction pipeline (teams 2–6) is classical SfM+MVS (COLMAP/OpenMVS/ODM-style),
  being optimized, not replaced. "Geometry gain" here targets triangulation-angle /
  baseline quality — not a feed-forward-transformer notion of coverage.
- Stack: Python + OpenCV + PyTorch.

## What failed before
A first attempt used object detection + ad hoc frame scoring. Object detection is
the wrong tool at aerial altitude — most objects fall under the ~32×32px detection
floor of standard detectors. Independent per-frame scoring also ignores redundancy:
it clusters "good" frames from the same few seconds and starves coverage elsewhere.

## Decided architecture (full detail in mechanism.md)
1. Robust frame extraction by real per-frame timestamp, not fps × duration.
2. Quality gate with a coverage-aware salvage rule — not a fixed universal threshold.
3. Semantic segmentation (not detection) for element extraction and dynamic-object
   masking.
4. A continuous solid-angle coverage function per scene region, built incrementally
   via homography chaining — not a discrete OBSERVED/WEAKLY/UNKNOWN grid.
5. Offline greedy submodular maximization over a pre-filtered candidate pool to pick
   the final frame subset. Proven ≥(1−1/e) (~63%) optimality guarantee.
6. Output: per-frame JSON manifest + one video-level coverage report.
7. A benchmark harness comparing full-frame vs. selected-subset reconstruction
   (time, point count, reprojection error) — this is the module's proof of value at
   demo time, not optional.

## Open / not yet decided
- Exact frame budget (k) and coverage-grid cell size — calibrate on real footage,
  don't guess.
- Exact handoff contract with team 2 — defaulting to frames + score breakdown +
  segmentation output as a JSON manifest (schema.md). Confirm before locking.
- Test footage: sourced from public drone-footage sites, no matching GPS/IMU.
