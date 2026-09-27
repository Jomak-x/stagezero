# Opt-in terrain descent follow-up

This follows merged PR #30 (`184eca0`). Terrain-aware movement remains **off by default**. Turning it off restores the separate ordinary ARDY take. No ordinary generation or rendering code changes in this follow-up.

## What changed

- Terrain planning wraps yaw across ±π instead of clamping a turn at the boundary.
- Explicit heading assistance uses the upcoming walking direction for a stationary lead-in pivot. Native pelvis overshoot cannot reverse the intended contact turn. Native arrays remain untouched; ambiguous 180° turns still reject.
- Descending feet finish horizontal travel before their final lowering sample. Failed swept heel clearance retries the existing bounded clearance solver, then reruns every quality gate. Missing support returns rejection diagnostics.

No acceptance limits were widened: 35° maximum local leg rotation per frame, 110° knee flexion, 24 cm lift, 8 mm penetration, and the existing slip/body proxy checks still apply.

## Fresh normal-app verification

All commands below were submitted through the normal Studio UI using the real Core GPU backend and real prompt encoder. Videos replay the exact saved UI results at 20 fps with a following camera; they do not record the UI controls or GPU wait time.

| Scenario | Commands and result | Evidence |
|---|---|---|
| Industrial route continuation | Reopen the prior 480-frame industrial entry take; `walk down loading exit steps` → 640 frames; separately `walk 1 metre forward` → 680; save/reopen; `walk 0.5 metres forward` → 720. | [12-second new-motion video](descent-and-repeat.mp4), [640-frame project](industrial-descent.core.stagezero.npz), [680-frame project](industrial-first-repeat.core.stagezero.npz), [720-frame project](industrial-descent-repeated.core.stagezero.npz) |
| Rotated and translated scene | Fresh actor at X 10.9924231865, Z -8.1490969664, yaw -2.4958208304; `walk down loading exit steps` → 80 frames. Geometry rotated 37° and translated by (7, 0, 4); procedural crates converted to equivalent custom boxes because procedural props cannot rotate. | [4-second video](rotated-descent.mp4), [scene](rotated-industrial.scene.json), [project](rotated-descent.core.stagezero.npz) |

Both videos were inspected at normal speed; capture contact sheets are included. The first clip is frames 480–719 of the 720-frame capture (24–36 seconds). The rotated clip is the complete 80-frame take. Neither claims an entirely new full rotated temple-to-gate GPU run.

[Prefix audit](prefix-audit.json) verifies exact native positions, rotations, features and assisted rig positions, rotations and stance arrays across 480 → 640 → 680 → 720. The starting 480-frame archive is [the original normal-app industrial project](../terrain-assisted/normal-app/industrial-complete.core.stagezero.npz). The audit contains historical local source paths; the archives are bundled here under descriptive names.

[Rotated audit](rotated-audit.json) independently recomputes rig FK, knee angles, local leg frame steps, native-root XZ preservation and asset provenance. The rotated take has 34.51° maximum local leg frame step and 96.98° maximum knee bend. Stored solver reports show zero actual/swept sole penetration and no unsupported samples. Collision metrics in these audits are recorded solver results, **not independent collision recomputation**. The body check uses a finite 6 cm proxy, not complete skin collision or physical dynamics.

Archive fields `accepted=false` and `visual_review=pending` are preserved as saved by the app. They are promotion/review metadata, not numerical failure flags; normal-speed review is recorded here rather than modifying exact archives.

## Regression coverage and reproduction

Run `PYTHONPATH=.:vendor/ardy python -m unittest discover -p 'test_*.py'` with the project dependencies installed. Added tests cover both yaw boundary crossings, exact π request validation, a widened/rotated/translated/elevated route through ascent/bridge/gate/descent plus repeated commands, route-directed pivots, ambiguous/short pivots, descending heel clearance, and missing-support diagnostics. The final suite output is [tests.txt](tests.txt).

To replay a project in the normal app, launch `director_viewer.py --reference-only --core-project <archive> --port <free-port> --environment none`. To generate more commands, also configure the existing authenticated Core backend URL/token file. Start the fresh rotated scene with `--objects review/terrain-followup/rotated-industrial.scene.json`, enable Terrain-aware movement, and use the start coordinates above. Credentials are intentionally not bundled.

## Supported boundaries

The reusable geometry planner handles suitable authored static support, shallow stairs, bridges, configured automatic doors and repeated bounded spatial commands. Ordinary ARDY remains separate. The assisted Human17 display changes facing, feet, legs and pelvis height while preserving native history.

Descent and difficult pivots remain experimental beyond the tested routes. Steep/irregular stairs, ambiguous floors, insufficient clearance, abrupt/half-turn pivots or unsuitable native output can reject safely. Jumping, ladders, climbing, hand-operated doors, object manipulation, moving platforms, arbitrary rigs and multi-actor terrain interaction remain unsupported. This does not make every generated background automatically traversable.
