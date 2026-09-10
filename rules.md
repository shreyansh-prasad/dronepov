# Rules — Scope Locks for This Module

Follow these without exception. If a task seems to require breaking one, stop and
flag it in progress.md — don't silently reinterpret the rule.

1. **Stack**: Python + OpenCV + PyTorch only. No cloud APIs, no multi-GPU
   assumptions. Must run on a single consumer laptop GPU.
2. **No GPS/IMU dependency in the default path.** Any feature needing telemetry
   must have a visual-only fallback that is the primary path, not a bolt-on for
   "if telemetry missing."
3. **No object detection for element extraction.** Use semantic segmentation only.
   Do not reintroduce bounding-box detectors for cars/people/buildings.
4. **No independent per-frame scoring.** Frame value is always a marginal gain
   against the currently selected set (mechanism.md Stage 5). Scoring frames
   independently and top-K'ing them is a rule violation, not a valid shortcut.
5. **No fixed universal quality threshold for the hard gate.** The gate must
   consult current coverage state before permanently rejecting a frame (the
   salvage rule, mechanism.md Stage 1).
6. **No real-time/streaming architecture.** This module is offline. Do not add
   streaming submodular maximization or single-pass/causal constraints — they
   solve a problem this module doesn't have and only lose approximation quality.
7. **Module boundary**: video file (+ optional GPS/IMU/flight metadata) in; JSON
   manifest + coverage report out. Do not implement or modify reconstruction/
   SfM/MVS code — that belongs to teams 2–6.
8. **No hardcoded magic constants** (blur threshold, k, grid cell size,
   triangulation-angle target). Every such value is a named, documented,
   configurable parameter, marked `# CALIBRATE ON REAL FOOTAGE` in code.
9. **Manifest schema changes** require updating schema.md in the same change — no
   silent field additions/removals.
10. **Don't gold-plate.** This is a hackathon module with a benchmark deliverable
    due. Between "theoretically better" and "ships and is measurable," ship the
    measurable one and log the gap in progress.md.
