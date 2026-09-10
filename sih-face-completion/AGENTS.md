# Shared Agent Standards

## Data Conventions
- Mesh interchange: OBJ or glTF, with an explicit per-vertex/per-face provenance array (OBSERVED / SYMMETRY_DERIVED / GENERATED) — never export a mesh without it
- Coordinate frame: match the terrain/world frame handed off by the reconstruction team exactly — never re-origin, rescale, or re-orient without an explicit, logged transform
- Config: use dataclasses or Pydantic models for pipeline parameters (symmetry confidence threshold, hole-size cutoff, etc.) — never raw dicts for anything read in more than one place

## Error Handling
- Never fail silently on an object — if it can't be processed (no symmetry found, completion net fails, corrupt mesh), log the object ID and skip it, don't crash the batch
- Always log with context: object ID, pipeline stage, threshold crossed
- Never emit a completed mesh without its provenance tags attached

## Python
- Type hints required on all function signatures
- No bare `except:` — catch specific exceptions
- No `print()` in pipeline code — use the `logging` module
- Prefer NumPy/vectorized operations over per-vertex Python loops
