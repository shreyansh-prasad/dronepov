# Project Context

> Auto-maintained by /sync workflow. Do not edit manually.

## Project Identity
- Name: sih-face-completion
- Parent project: SIH25158 (single-pass drone video 3D reconstruction)
- Sibling module: ../frame-ranking (separate Antigravity project — do not merge context)
- Stack: Python 3.x, mesh library TBD (candidates: trimesh, Open3D), PyTorch (fallback completion net), NumPy
- Type: Offline CV/geometry batch pipeline — no UI, no API, no server
- Package manager: pip (TBD: plain pip vs poetry)

## Architecture
- Entry point: TBD (e.g. run_completion.py)
- IO layer: io/ — loads pre-segmented per-object mesh + terrain placement + reference images from the reconstruction team's handoff
- Detection layer: detection/ — hole/no-data detection, symmetry-plane detection, confidence scoring
- Completion layer: completion/ — symmetry reflection (geometry + texture), fallback point-cloud completion net, 2D texture-inpainting fallback
- Output layer: output/ — provenance tagging (OBSERVED / SYMMETRY_DERIVED / GENERATED), export to downstream format
- Tests: TBD

## Layer Map (What Files Belong to What Layer)
### IO Layer (never touch during detection/completion/output work)
- io/

### Detection Layer (never touch during IO/completion/output work)
- detection/

### Completion Layer (never touch during IO/detection/output work)
- completion/

### Output Layer (never touch during IO/detection/completion work)
- output/

## Active Decisions
| Decision | Reason | Date |
|----------|--------|------|
| Symmetry reflection is the primary completion method, not pure generative | Only method that preserves real dimensional accuracy from captured data | 2026-09-10 |
| Generative fallback (point-cloud completion net + 2D inpainting) used only for asymmetric residual gaps | Accuracy requirement rules out generative-only completion | 2026-09-10 |
| Every filled region carries a provenance tag: OBSERVED / SYMMETRY_DERIVED / GENERATED | Downstream consumers must know which faces are trustworthy for measurement | 2026-09-10 |
| Object instances arrive pre-segmented from the reconstruction team | Confirmed handoff — this module does not do object segmentation | 2026-09-10 |
| No-data zones are self-detected, not flagged by upstream | Confirmed — incoming mesh has no gap annotations | 2026-09-10 |
| Symmetry acceptance is threshold-gated (self-alignment/ICP score) | Below-threshold symmetry falls through to GENERATED tag instead of being trusted | 2026-09-10 |
| Symmetry detected per rigid sub-segment, not per whole object | Avoids mirroring compound/asymmetric shapes (L-shaped buildings, racked cars) incorrectly | 2026-09-10 |

## Current Functionality (Stable — Do Not Break)
- [ ] Nothing built yet — greenfield module

## In Progress
- Hole / no-data detection on incoming per-object mesh — not started
- Symmetry-plane detection + confidence scoring (per sub-segment) — not started
- Symmetry reflection: geometry + texture — not started
- Occluded-side partial-data validation (use stray low-quality frames as a check against mirrored prediction) — not started
- Fallback completion net integration (generic rigid-object point cloud completion) — not started
- 2D texture-inpainting fallback for texture-only gaps — not started
- Provenance tagging + export format — not started

## Known Issues
- Symmetry detection cannot self-verify on a single occluded object — a genuinely asymmetric object with a high accidental self-alignment score could still be mirrored wrong. Confidence threshold value itself is not yet chosen.
- Exact pretrained checkpoint for the fallback point-cloud completion net not yet chosen (TBD)
- Exact hole-detection / mesh library not yet chosen — candidates: trimesh, PyMeshLab (TBD)
- No confirmed access to per-vertex observation-count metadata from the reconstruction team's mesh — if unavailable, "weakly observed but present" regions can't be distinguished from genuinely solid ones (TBD, needs a decision with the reconstruction team)

## Installed Packages
- None yet

## Last Updated
2026-09-10 — bootstrap
