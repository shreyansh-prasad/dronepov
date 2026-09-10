---
trigger: always_on
description: Hard layer boundaries — prevents cross-layer contamination
---

# Layer Lock Rules

Every task has a [LAYER] header. Obey it absolutely.

## If LAYER: IO
Files you MAY touch: io/ (mesh loaders, image loaders, upstream format parsers)
Files you MUST NEVER touch: detection/, completion/, output/
If an IO change requires a detection or completion change: STOP. Report it. Do not proceed.

## If LAYER: Detection
Files you MAY touch: detection/ (hole detection, symmetry-plane detection, confidence scoring)
Files you MUST NEVER touch: io/, completion/, output/
If a detection change requires a completion change: STOP. Report it. Do not proceed.

## If LAYER: Completion
Files you MAY touch: completion/ (symmetry reflection, fallback completion net, texture inpainting)
Files you MUST NEVER touch: io/, detection/, output/

## If LAYER: Output
Files you MAY touch: output/ (provenance tagging, export)
Files you MUST NEVER touch: io/, detection/, completion/

## If LAYER: Both
Only touch files explicitly listed in the task's Context section.
Any file not listed is off-limits, regardless of layer.

## Why This Exists
This module mixes cheap deterministic geometry (symmetry reflection) with expensive learned fallbacks (completion net). An unscoped change that touches both can quietly shift output from SYMMETRY_DERIVED (trustworthy) to GENERATED-quality without anyone noticing — that is the specific failure this rule blocks.
