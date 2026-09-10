"""
Completion dispatcher.

Routes each ObjectInstance to the correct completion strategy:
  - "symmetry" → SymmetryCompletion (buildings, vehicles)
  - "inpaint"  → InpaintCompletion  (trees, humans)
  - "none"     → skip

If symmetry completion fails (no plane found or ICP below threshold),
it automatically falls through to InpaintCompletion as a safety net.
"""
from __future__ import annotations

import logging

from scene_io.scene import ObjectInstance
from .symmetry_completion import SymmetryCompletion
from .inpaint_completion  import InpaintCompletion

logger = logging.getLogger(__name__)


class CompletionDispatcher:
    """
    Orchestrates all completion strategies across all instances in a scene.

    Usage:
        dispatcher = CompletionDispatcher()
        dispatcher.run(instances)
    """

    def __init__(self) -> None:
        self._symmetry = SymmetryCompletion()
        self._inpaint  = InpaintCompletion()

    def run(self, instances: list[ObjectInstance]) -> None:
        """Apply the appropriate completion strategy to each instance."""
        for inst in instances:
            if not inst.needs_completion:
                logger.info(
                    "[Instance %d | %s] Skipping completion (no holes or strategy=none).",
                    inst.instance_id, inst.semantic_name,
                )
                continue

            strategy = inst.completion_strategy
            self._apply(inst, strategy)

    def _apply(self, inst: ObjectInstance, strategy: str) -> None:
        try:
            if strategy == "symmetry":
                success = self._symmetry.complete(inst)
                if not success:
                    logger.info(
                        "[Instance %d] Symmetry failed → falling through to inpaint.",
                        inst.instance_id,
                    )
                    self._inpaint.complete(inst)

            elif strategy == "inpaint":
                self._inpaint.complete(inst)

            elif strategy == "none":
                pass  # intentional skip

            else:
                logger.warning(
                    "[Instance %d] Unknown strategy '%s' — skipping.",
                    inst.instance_id, strategy,
                )
        except Exception as exc:
            # AGENTS.md: never fail silently — log and skip the object, don't crash
            logger.error(
                "[Instance %d | %s] Completion failed with exception: %s — skipping.",
                inst.instance_id, inst.semantic_name, exc,
                exc_info=True,
            )
