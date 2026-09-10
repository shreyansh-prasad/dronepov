# Mechanism — Frame Selection & Ranking Pipeline

## Stage 0 — Extraction
- Decode via real per-frame presentation timestamps (PyAV or ffmpeg). Never use
  `duration × declared_fps` — real drone MP4s have variable frame rate / dropped
  frames.
- Build a downsampled proxy resolution for all analysis stages; keep full-res only
  for frames that survive to the final selected set.

## Stage 1 — Hard gate (reject) + salvage rule
Reject:
- Corrupt / undecodable frames.
- Blur below a Laplacian-variance floor, calibrated per-video/per-camera — don't
  reuse a published constant.
- Severe exposure clipping.
- Near-duplicate frames (near-zero apparent motion vs. last accepted frame — cheap
  ORB/optical-flow check).

**Salvage rule** (do not skip — this fixes "irreversible gate deletes the only
evidence"): before finalizing a rejection, check whether the frame is the only
candidate touching a currently under-covered region (Stage 3). If yes, don't
reject — accept it tagged `low_confidence: true`. Downstream robust bundle
adjustment (Huber/Cauchy loss) already discounts noisy observations; a tagged weak
frame degrades that region gracefully instead of creating a hole with zero
evidence. This makes the gate a function of current coverage state, not a fixed
universal cutoff.

Dynamic objects (vehicles, people, animals) and sky are NOT frame-level gates —
mask them within the frame (Stage 2 output), don't reject the whole frame.

## Stage 2 — Semantic segmentation (replaces object detection)
- Model: UNetFormer (ResNet18 encoder) or equivalent lightweight segmenter,
  pretrained on UAVid classes: building, road, tree, low vegetation, static car,
  moving car, human, clutter.
- Fine-tune on your own footage if time allows; do not train from scratch on a
  hackathon clock.
- Outputs feed:
  - Stage 1's dynamic-object/sky masking.
  - Mission-value weighting in Stage 5 (e.g. weight "damaged structure"/"building"
    pixels above empty terrain — mission-specific, tunable, not a fixed formula).

## Stage 3 — Coverage state
- Maintain a coarse spatial grid over the scene. For each cell, track a
  **continuous** solid-angle coverage value contributed by all accepted frames
  observing it (model: view-angle disks per surface point, weighted for viewing
  diversity and fronto-parallel proximity) — not a 4-state categorical label.
  OBSERVED/WEAKLY OBSERVED/UNKNOWN, if kept for a demo visualization, are just
  thresholds read off this continuous value — not the underlying data structure.
- Update incrementally via homography chaining between consecutive accepted
  frames (no GPS/IMU required).

## Stage 4 — Candidate shortlist (cheap pre-filter)
- From gated frames, compute a cheap novelty proxy: ORB/FLANN keypoint match count
  or homography inlier ratio vs. the nearest already-selected frame, on
  downsampled images.
- Shortlist ~150–250 candidates from the full gated set (not raw ~900) — keeps
  Stage 5 tractable on a laptop GPU.

## Stage 5 — Final selection: offline greedy submodular maximization
- On the shortlisted pool, run a sparse pose estimate (essential-matrix /
  incremental SfM) to get real triangulation angles.
- `USEFULNESS(frame | selected_set) = marginal_coverage_gain(frame | selected_set)
  + mission_value(frame) − processing_cost(frame)`
  - `marginal_coverage_gain` = increase in Stage-3 coverage value if this frame is
    added to the currently selected set. Replaces the earlier separate
    geometry_gain / uncertainty_reduction split — same math, one term.
  - `mission_value` = semantic-class priority weighting from Stage 2 (tunable per
    mission type).
  - `processing_cost` = proxy for added bundle-adjustment cost (roughly constant
    per frame; matters mainly if k is large).
- Algorithm: classical greedy — at each step, recompute USEFULNESS for every
  remaining candidate against the current selected set, pick the max, add it,
  repeat until frame budget k is reached. This is full offline greedy, not
  streaming — the whole candidate pool is in memory, so it earns the stronger
  (1−1/e) ≈ 63% optimality guarantee instead of the 1/2 bound that streaming
  algorithms (e.g. Sieve-Streaming) are stuck with when they can't look ahead.

## Stage 6 — Output
- Per-frame JSON manifest: frame id, timestamp, USEFULNESS score + breakdown,
  semantic class histogram, mask file path, `low_confidence` flag. Schema:
  schema.md.
- One video-level coverage report: % of scene at OBSERVED/WEAKLY OBSERVED/UNKNOWN
  (thresholded from the continuous coverage value), per-class pixel totals.

## Stage 7 — Benchmark (required deliverable, not optional)
- Run the team-2 reconstruction pipeline twice: once on all gated frames, once on
  the selected subset.
- Report: wall-clock time, point count, mean reprojection error, side by side.
  This table is the module's proof of value at demo time.
