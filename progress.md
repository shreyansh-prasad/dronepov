# Progress

Status legend: [ ] not started · [~] in progress · [x] done

## Pipeline
- [x] Stage 0 — Extraction (real PTS decode)
- [x] Stage 1 — Hard gate + salvage rule
- [x] Stage 2 — Segmentation (UNetFormer/UAVid)
- [x] Stage 3 — Coverage state (continuous, solid-angle)
- [x] Stage 4 — Candidate shortlist (cheap novelty proxy)
- [x] Stage 5 — Offline greedy selection
- [x] Stage 6 — Manifest + report output
- [x] Stage 7 — Benchmark harness

## Calibration (blocked on real footage)
- [ ] Blur threshold
- [ ] Grid cell size
- [ ] Frame budget k
- [ ] Triangulation-angle target range

## Open decisions
- [ ] Team-2 handoff format confirmed (currently: default JSON manifest,
      unconfirmed)
- [ ] Test footage sourced + verified decodable

## Changelog
(append dated entries here as work happens — don't rewrite history)
- 2026-09-08: Setup end-to-end pipeline implementation across all 8 stages.
- 2026-09-08: Stage 0: Added real PTS decoding using PyAV.
- 2026-09-08: Stage 1/3/4: Built Quality Gate checking blur/exposure and salvage rule backed by homography continuous coverage tracker.
- 2026-09-08: Stage 2: Mapped torch segmentation fallback for UAVid.
- 2026-09-08: Stage 5: Implemented offline greedy selection function (capped marginal gain) and verified via explicit duplicates unit test.
- 2026-09-08: Stage 6/7: Exported JSON manifest per schema and generated benchmark comparison harness mock.
