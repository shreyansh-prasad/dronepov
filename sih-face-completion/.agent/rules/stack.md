---
trigger: always_on
description: Stack enforcement rules
---

# Stack Rules
- Read @CONTEXT.md for the current stack before every task
- Python 3.x, type hints on every function signature
- Mesh/geometry library: use what's already installed — check CONTEXT.md before adding a new one
- Fallback completion net: use the pretrained checkpoint recorded in CONTEXT.md — do not swap models without explicit approval
- Logging: use the `logging` module — never `print()` in pipeline code
- Do not install a new package without explicit approval
- This pipeline runs offline on a local GPU laptop — do not add cloud-API-dependent steps (hosted diffusion APIs, remote inference) without explicit approval
