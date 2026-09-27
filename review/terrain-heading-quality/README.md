# Smoother heading at terrain continuation starts

This follow-up reduces a brief assisted-facing wobble when a new terrain action begins with a short stationary interval. Terrain remains **opt-in and off by default**. Ordinary ARDY generation and rendering are unchanged.

The existing curve fades the inherited heading offset while following the new route heading immediately. The candidate also fades the new heading target over the first 0.6 seconds. It is used only when walking starts inside that window, no intervening pivot contact is planned, and neither peak yaw speed nor peak yaw acceleration increases against the existing curve. The comparison includes samples after the fade, so moving a spike to the outgoing join does not qualify. Otherwise the existing curve is retained.

The incoming comparison assumes a settled heading: it prepends two identical inherited-heading samples. The API supplies one previous pose, not the preceding angular velocity. This is therefore a bounded improvement for settled continuation starts, not a guarantee about acceleration across every possible moving or pivoting boundary. Long intentional pivots and immediate walking retain their existing handling.

## Matched replay comparisons

Each comparison reuses the same native motion and committed presentation prefix; only the appended presentation is solved again. These are CPU presentation comparisons, not fresh GPU generations. Speed is in degrees/second and acceleration in degrees/second².

| Continuation | Startup yaw speed, parent → candidate | Startup yaw acceleration, parent → candidate | Join yaw speed, parent → candidate | Evidence |
| --- | --- | --- | --- | --- |
| Partial ascent followed by a short walk, frames 40–79 | 126.95 → 36.35 | 1366.47 → 432.55 | 68.32 → 1.66 | [Metrics](partial.json), [project](partial.core.stagezero.npz) |
| Industrial repeated walk, frames 680–719 | 84.49 → 54.53 | 1688.26 → 654.39 | 84.41 → 2.01 | [Metrics](industrial.json), [project](industrial.core.stagezero.npz) |
| Temple first continuation, frames 360–399 | 105.19 → 104.96 | 1854.98 → 1179.14 | 35.12 → 0.82 | [Metrics](temple-first.json), [matched parent](temple-first-current-parent.json), [project](temple-first.core.stagezero.npz) |

The temple row uses `current_baseline`, which reruns the parent solver on the same input. Its older archived `before` curve came from an earlier solver and is not the matched baseline for this change. In particular, the candidate does not improve every metric against that older archive.

[Independent world-rotation derivatives](torso-world-derivatives.json) compare the root and two torso joints against the matched parent. Peak angular speeds do not increase in these three cases, and their peak angular accelerations decrease. All listed parent and candidate solves pass their existing gates. These measured cases do not establish a general guarantee for every upper-body rotation.

The [long-pivot comparison](long-pivot.json), frames 480–639 of the industrial route, does not activate the fade. Its presentation positions and rotations are exactly unchanged from the current parent solve.

## Exact data and physical checks

For the partial and industrial archives, native positions, rotations, and features are exactly equal to their original saved projects. Their first 40 and 680 presentation frames respectively—including positions, rotations, and stance—are also exactly preserved. The new presentation preserves native root XZ and relative upper-body joint rotations. World-facing assistance can change the appended body's world rotations and the resulting leg solve; it does not rewrite native motion or previously committed frames.

The contact planner, its timing, and physical acceptance limits are unchanged. The candidate still passes the existing 24cm swing-lift, 8mm penetration, 5cm/s planted-foot slip, 110° knee-flexion, and 35°/frame local-leg-rotation limits, plus the route/support and finite body checks. Recorded sole, sweep, and body metrics are solver evidence, not a new independent collision certification. Archive promotion fields remain `accepted=false` and `visual_review=pending`.

## Playback and fresh normal-app verification

The comparison videos show **parent on the left, candidate on the right**, at **1× speed**:

- [Partial continuation comparison](partial-before-after.mp4); [full 4-second candidate replay](partial-full.mp4), [capture metadata](partial-capture.json), [contact sheet](partial-contact-sheet.png).
- [Industrial continuation comparison](industrial-before-after.mp4); [full 36-second candidate replay](industrial-full.mp4), [capture metadata](industrial-capture.json), [contact sheet](industrial-contact-sheet.png).

The full replays render the saved Human17 presentation at 20 fps with a following camera, without additional retargeting or IK. They show playback, not generation wait time or UI controls.

In the full industrial replay, the door occludes the actor around 14–17 seconds. The affected continuation at 34–36 seconds remains visible.

Normal-speed review around the continuation joins found a modest improvement in the gentleness of the torso turn, with the stepping pattern unchanged. This visual assessment is recorded here; it does not alter the archive's promotion metadata.

A separate normal Studio run reopened the saved 40-frame partial ascent and submitted a new continuation through the terrain controls using the real Core backend. The resulting [80-frame project](fresh-continuation.core.stagezero.npz) contains the saved 40-frame prefix and 40 newly generated frames. The [audit](fresh-audit.json) confirms exact native and presentation prefix preservation, matching render assets, exact native root XZ, and no recorded numerical rejections. Recomputed maxima are 23.12° per local leg rotation step and 84.28° knee bend. See the [normal-app screenshot](fresh-normal-ui.png). This was a fresh continuation, not an entirely new 80-frame generation.

## Regression verification

The [full local suite output](tests.txt) records **1,593 tests run, OK, with 2 skipped**. Four new heading tests cover the startup improvement, rejection of a worse outgoing join, unchanged long pivots/immediate walks/contact turns, and equivalent headings across the ±π boundary. Run `PYTHONPATH=.:vendor/ardy python -m unittest discover -p 'test_*.py'`; local HTTP tests require loopback access.

To replay an exact project, run `director_viewer.py --reference-only --core-project <project> --port <free-port> --environment none`. See [terrain setup and limitations](../../docs/TERRAIN-AWARE.md) for normal-app generation. This change does not broaden terrain geometry, action, rig, or physical-simulation support.
