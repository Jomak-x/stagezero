# Reusable terrain-assisted traversal

**Normal Studio integration:** see [fresh normal-app generation, repeated commands, save/load, and ordinary-motion preservation](normal-app/README.md), with [usage and interaction limits](../../docs/TERRAIN-AWARE.md). The older separate-viewer evidence below is retained for the development record.

The complete temple sequence now runs end to end: **stairs → bridge → automatic gate lift → enter**. The saved 360-frame, 20 fps route uses real GPU ARDY motion and a separate actual-character contact solve. Both the temple and a wider Copper observatory background pass contact/body/gate checks and have complete normal-speed videos. Ordinary Core generation and its renderer remain unchanged. The new viewer never submits assisted transforms as ARDY history.

- [Temple video](full-route/performance.mp4), [contact sheet](full-route/contact-sheet.png), [visual review](full-route/visual-review.json).
- [Copper observatory video](observatory/performance.mp4), [contact sheet](observatory/contact-sheet.png), [comparison provenance](observatory/observatory.report.json).
- [Native GPU evidence](full-route/manifest.json) and [separate paired replay](full-route/proof.assisted.npz).

The Copper comparison reuses the same native take in a renamed, recolored, wider scene; it is not a second GPU generation or a different navigation topology. A rotated/translated scene and different initial heading are additionally covered by CPU planning tests.

## What the new evidence establishes

One new native seed-33 trial used five real 4 cm risers, 50 cm treads and 2 m width. Native sole penetration still reached about 4 cm. The previous bounded contact adapter also rejected this clip. Those results are retained under `shallow-native/`; they are not successes.

Solving only Core27 feet was insufficient: retargeting that candidate to `civilian.glb` produced 5.66 cm of actual shoe penetration. The rendered leg chain is about 70 cm, versus about 87 cm in Core, and its shoe is about 31 cm long, versus the previous 20 cm proxy. See `rig-proof/rejected-core-retarget-mesh.json`.

`terrain_assisted_rig.py` now retargets native motion once, then plans contacts and solves the actual 17-bone character rig. `terrain_assisted_renderer.py` draws those transforms directly, without another retargeting or foot correction pass. Native Core positions, rotations and all 330 feature channels remain separate. Native root XZ and the retargeted upper-body rotations remain unchanged; cadence, feet, legs and pelvis Y are explicitly assisted.

The complete route has zero measured actual sole-vertex and swept-envelope penetration, zero unsupported samples, and effectively zero flat-stance vertex slip. Its largest pelvis correction is 24.8 cm. Knee flexion stays below the unchanged 110-degree limit and adjacent local leg rotation changes below 35 degrees. These bounds are enforced while solving, rather than hidden by a renderer adjustment. This is not a cosmetic correction or native stair gait. Contact measurements and finite swept proxies do not certify dynamics or every possible mesh collision.

Both videos were inspected at normal speed and through their full-route contact sheets. Temple replay was also played and sought backward/forward across the gate trigger. The lower-body cadence is visibly procedural; native upper-body movement remains. `visual_review=reviewed_at_1x` records that inspection; `accepted=false` preserves the distinction from the user's subjective style approval or promotion into the ordinary Studio path.

## Reuse and replay

Using the normal project Python environment with its dependencies and built Studio client:

```sh
PYTHONPATH=vendor/ardy python terrain_assisted_viewer.py \
  --project review/terrain-assisted/full-route/proof.assisted.npz --port 24997
```

The viewer has playback, seeking, save/load, editable starting X/Z/yaw, cancellation and an explicit **new take** action. New generation requires `--backend` and `--token-path` for a compatible Core service; replay needs neither. Each new take starts from its configured placement; continuation from a native-only prefix is rejected because its actual assisted boundary is missing. The completed native route is checkpointed before assistance; a CPU contact failure or cancellation retains that recovery archive. This session verified GPU generation through the official runtime adapter and replay through the viewer, not the viewer's Generate button against a live HTTP backend.

`traversal_kit.traversable_temple_scene()` and `traversable_observatory_scene()` supply real shallow stairs, a connected bridge, an automatic lifting gate and a supported interior. Custom Scene3 backgrounds can be supplied with `--scene`; the route planner derives support and obstacles from rendered geometry, not temple IDs. Give walkable objects distinct names and supply actual support surfaces: an apparent bridge painted into a background is insufficient. The tested templates use 4 cm risers, 50 cm treads and at least 2 m path width. Missing support, an absent bridge, ambiguous targets, closed passages and infeasible footprints reject. Steeper stairs, arbitrary pivots and new topologies are not established by these two examples.

The command used by the fixture tests is:

> walk up the shallow temple stairs, cross the bridge, open temple gate, and enter

`generate_native_terrain_commands` buffers a complete native sequence privately and records measured action boundaries. `assist_native_terrain_result` can then solve the saved result repeatedly on CPU without more inference. Native and assisted NPZ formats are distinct, bounded and preserve their own provenance and character asset hash.

## Verification and reproduction

The four-action sequence passes a CPU integration test with an injected client: 440 frames, 11 horizons, measured stair/bridge arrival, observed gate lift and entry. That test proves orchestration, **not real ARDY motion**.

The user approved uploading the 37 required Python files and scene to their first RunPod. The native-only GPU run in `experiments/core_assisted_temple_trial.py` completed all four actions: 360 frames, nine horizons, seed 33. It used the documented cached stand/walk text embeddings. It checkpoints every native horizon before later checks and saves the complete native route, so assistance failures do not require repeated inference. CPU fixes addressed compressed restart swings, a skipped short-approach contact and a missing knee-feasibility optimizer bound; no quality threshold was widened.

The final focused run passed 94 tests across route planning, transformed/renamed scenes, missing and blocked geometry, elevated gate reactions, native adapter/history/cancellation, contact solving, archive bounds/provenance and ordinary session behavior. The full repository test suite was not completed. Normal Studio rendering and generation were not replaced.

To reproduce the second background comparison without a GPU:

```sh
PYTHONPATH=.:vendor/ardy python -m experiments.observatory_background_reuse --help
```

To capture a paired replay without another retarget or IK pass:

```sh
PYTHONPATH=.:vendor/ardy python experiments/capture_terrain_assisted.py \
  --project review/terrain-assisted/full-route/proof.assisted.npz \
  --output .runtime/terrain-capture --port 24993 --follow-camera
```

Neon GPT-6 Astra's bounded contact/rig reviews are in `research/`. They supplied engineering guidance; actual implementation and measurements are checked locally.
