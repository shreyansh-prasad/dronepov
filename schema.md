# Manifest Schema (v0 — draft, confirm with team 2 before locking)

## Per-frame manifest entry
```json
{
  "frame_id": "string",
  "source_timestamp_ms": "number — from real PTS, not fps*index",
  "usefulness_score": "number",
  "score_breakdown": {
    "marginal_coverage_gain": "number",
    "mission_value": "number",
    "processing_cost": "number"
  },
  "low_confidence": "boolean — true if salvaged past the quality gate",
  "semantic_histogram": { "<class_name>": "pixel_fraction (0-1)" },
  "mask_path": "string — path to saved segmentation mask",
  "gate_reason": "string|null — populated if rejected, null if accepted"
}
```

## Video-level coverage report
```json
{
  "total_frames_input": "number",
  "total_frames_selected": "number",
  "coverage_by_status": {
    "observed_pct": "number",
    "weakly_observed_pct": "number",
    "unknown_pct": "number"
  },
  "class_pixel_totals": { "<class_name>": "number" }
}
```

Thresholds for observed/weakly_observed/unknown are read off the continuous
per-cell coverage value in mechanism.md Stage 3 — fill in once calibrated:
- observed threshold: TBD
- weakly_observed threshold: TBD
