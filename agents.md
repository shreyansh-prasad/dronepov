# Agents — Antigravity Multi-Agent Split

Each agent should be runnable and testable in isolation before wiring together.

## 1. Extraction Agent
- Input: raw drone video file.
- Output: frames + real per-frame timestamps + downsampled proxy set.
- Success: frame count/timing match actual decoded PTS, not fps×duration, on a
  video with known variable frame rate.
- Failure modes: dropped-frame videos silently mis-timed; a corrupt frame crashing
  the whole run instead of being skipped and logged.

## 2. Quality-Gate Agent
- Input: extracted frames + current coverage state (from Agent 4).
- Output: gated frame set with accept/reject/low_confidence per frame + reason.
- Success: blur/exposure rejects calibrated per-video; salvage rule correctly
  overrides rejection when the frame is the sole observer of an under-covered
  region.
- Failure modes: gate running before any coverage state exists (needs an initial
  coarse pass first — wire this dependency explicitly); salvage rule never firing
  because coverage state isn't actually being checked.

## 3. Segmentation Agent
- Input: gated frames.
- Output: per-frame semantic mask (UAVid classes), dynamic-object mask, sky mask.
- Success: mask covers full frame; inference runs within per-frame time budget on
  target GPU.
- Failure modes: model trained/evaluated on a different altitude range than your
  footage — published mIoU won't transfer without checking on your own data.

## 4. Coverage/Ranking Agent
- Input: gated + segmented frames.
- Output: candidate shortlist, per-region continuous coverage values, final
  selected frame subset with USEFULNESS breakdown.
- Success: greedy selection re-evaluates marginal gain against the CURRENT
  selected set at every step (not a single static score pass); temporal order of
  input frames doesn't change which frames get selected for a given coverage
  target.
- Failure modes: highest-risk agent — reverting to independent scoring under time
  pressure is the likely failure. Test explicitly: feed 3 near-duplicate
  high-quality frames + 1 unique-but-mediocre frame; correct behavior picks the
  unique one after the first duplicate, not all 3 duplicates.

## 5. Manifest/Report Agent
- Input: final selected frame set + all upstream metadata.
- Output: JSON manifest (schema.md) + coverage report.
- Success: schema.md and actual output match exactly; report percentages sum
  sensibly.
- Failure modes: schema drift from silent upstream field changes.

## 6. Benchmark Agent
- Input: gated frame set (all) + selected subset.
- Output: comparison table (time / point count / reprojection error) from running
  team-2's pipeline on both.
- Success: reproducible — same inputs give same comparison numbers on rerun.
- Failure modes: benchmark run once and reported without a rerun to check
  reproducibility; team-2's pipeline interface changes without this harness being
  updated.
