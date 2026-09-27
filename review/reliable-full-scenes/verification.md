# Full-scene recovery and editing verification

Tested against the existing G1 pod on 2026-09-26 using the user's exact sprint/stop/fall/get-up/dance/backflip prompt.

- Original saved take remained prone after its two-second get-up; subsequent dance and backflip segments also stayed near floor height.
- New full-scene run completed seven actions in 15.72 seconds. The recovery used a six-second estimate, reached the sustained upright geometry check at two seconds, and stopped there. Dancing ended upright. Per-action measurements are in `live-result.json`.
- Browser on isolated port 24931: clicking movement 4 opened Full scene → Refine with movement 4 selected and playhead unchanged at frame 0.
- Two live recovery edits reached the final backflip, which failed the backend motion-quality checks after bounded retries. Both preserved the complete original scene.
- Replacing the last movement with a wave succeeded. Undo restored the original backflip, duration, and frame-0 playhead.
- A physical 20-meter distance and stunt accuracy were not asserted; upright recovery checks are geometric proxies. Model-quality rejections remain possible and do not install partial edits.

The existing Tailscale platform was left running with its in-memory project intact. These changes were verified in an isolated preview.

Final validation: all 827 discovered Python tests passed after the scoped timing fix; `git diff --check` passed. Dependencies were initialized at the repository's pinned submodule revision.
