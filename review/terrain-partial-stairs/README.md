# Continue from partway along a staircase

Follow-up to merged PR #38 (`66761da`). Terrain remains **opt-in and off by default**. This changes only terrain route planning; ordinary ARDY generation and rendering are unchanged.

## Reproduced bug and fix

A supported actor on the penultimate approach tread at X=0, Z=-1.75 could not execute `walk up foundry approach stairs`: the planner incorrectly treated proximity to the top as having already passed it. The symmetric partial descent also rejected. A legal preceding relative movement command can end at either location.

The planner now recognizes a partial start only when its height matches an actual rendered tread of the selected stair object. It requires consecutive remaining treads through the destination. Starts on nearby decks do not gain this exception, terminal treads still reject as already arrived, and existing navigation/body-clearance checks remain intact. The validator's pre-existing treatment of backward observations is unchanged; this is not a new guarantee about arbitrary looping paths.

## Fresh normal-app tests

All generation used the real Core GPU backend and prompt encoder, submitted through normal Studio's terrain controls with the included [industrial scene](industrial.scene.json). No cached motion was used for these UI submissions.

| Test | Result | Exact evidence |
|---|---|---|
| Start X=0, Z=-1.75, yaw=π; `walk up foundry approach stairs` | 40-frame partial ascent accepted. | [Project](partial-ascent.core.stagezero.npz) |
| Separate `walk 0.5 metres forward` afterward | 80 frames total; first 40 native/display frames preserved exactly. | [Project](partial-ascent-repeated.core.stagezero.npz), [4-second video](partial-ascent-repeated.mp4), [audit](ascent-audit.json) |
| Start X=10.5, Z=-10.625, yaw=π; `walk down loading exit steps` | Fresh 40-frame partial descent accepted. | [Project](partial-descent.core.stagezero.npz), [2-second video](partial-descent.mp4), [audit](descent-audit.json) |
| Repeat the completed descent command | Rejected as already at destination; all native/display arrays remain exact. | [Project after rejection](partial-descent-after-rejection.core.stagezero.npz) |
| Turn terrain off, then back on | Ordinary empty session restored while off; terrain 40-frame take restored exactly when re-enabled. | [Project after toggle](partial-descent-after-toggle.core.stagezero.npz), [audit](descent-audit.json) |

Both videos replay the exact UI-generated archives at 20 fps with a following camera and were inspected at normal speed. Capture metadata retains its original paths. Videos do not show the UI controls or generation wait time. The app's saved `accepted=false` and `visual_review=pending` promotion metadata are preserved; visual review is recorded here.

Independent FK/angle/provenance audit: repeated ascent maximum local leg step 23.10°, knee bend 84.18°; descent maximum step 26.02°, knee bend 90.43°. Native root XZ remains exact. Collision/slip/body metrics in the audit are stored solver evidence, not independent collision recomputation. The existing body check is a finite proxy, not full physical simulation.

## Regression verification

Five new tests exercise repeated movement→stair continuation in both directions, a 37° rotated/translated/raised scene, second and penultimate treads, same-height off-flight support, wrong sides and terminal destinations. These are planner tests; the transformed partial routes were not separately GPU-generated in this follow-up.

Full local suite: **1,589 tests passed, 2 skipped**; [output](tests.txt). Run `PYTHONPATH=.:vendor/ardy python -m unittest discover -p 'test_*.py'`. The local HTTP tests require loopback socket access. Python compilation and whitespace checks also pass.

To replay, launch `director_viewer.py --reference-only --core-project <project> --port <free-port> --environment none`. For fresh generation, launch with `--objects review/terrain-partial-stairs/industrial.scene.json` plus your existing authenticated Core backend configuration, set the listed start coordinates in the normal app, and explicitly start the terrain actor.

This fixes route continuation on suitable authored static stairs. It does not add jumping, climbing, manipulation, arbitrary rigs or automatic traversal of every background. Descent and difficult pivots retain their experimental limits and may reject unsuitable native output. See [the supported scope](../../docs/TERRAIN-AWARE.md).
