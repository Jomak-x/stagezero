# Full-scene popup verification

The isolated Studio browser preview was exercised against the configured live G1 Pod and text gateway.

- Submitted one athletic-performance prompt with a 60-second duration in the popup.
- Generated a 1,500-frame take at 25 fps with 12 separately editable movements.
- Closed and reopened the popup; the draft and scene job remained available.
- Replaced movement 11 through Refine; the selected movement and movement 12 regenerated as one continuous suffix.
- Saved both versions and compared their arrays: the first 1,250 frames (50 seconds) were identical, and the final 250 frames changed. The following movement retained its original instruction.
- Used the popup's Undo control, saved again, and confirmed all original motion, position, rotation arrays and movement segments were restored exactly.

`live-verification.json` records the comparison results. The motion archives and credentials remain private runtime files.

The initial live attempt exposed a quality-error classification mismatch. Regression tests now cover the Pod's exact prefixed error response, bounded quality-only retries, and cancellation during retry. Backend failure never installs a partial scene.

Reopened the refined project after restarting the Studio and used **Refine → Use current take** to select and regenerate its movements. The popup displayed live movement-update progress and the completion result. Final affected-feature regression suite: 131 tests passed. The existing browser client also built successfully.
