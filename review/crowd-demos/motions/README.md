# Crowd demo motion atlas

Eight GPU affine clips reuse the Xbot rig and the established 22-joint native affine solver:

| Index | Clip | Source | Playback |
|---|---|---|---|
| 0 | walk-relaxed | Existing reviewed Core source | Navigation-distance loop |
| 1 | walk-brisk | Existing reviewed Core source | Navigation-distance loop |
| 2 | idle | Existing generated Core wait | Loop |
| 3 | wave | Fresh Core, seed 941 | 1.933 s; blend out |
| 4 | look | Fresh Core, seed 942 | Subtle idle loop |
| 5 | listen | Fresh Core, seed 943 | Conversational hands gesture, 1.933 s; blend out |
| 6–7 | greeting-role-0 / greeting-role-1 | Existing InterGen pair, seed 49 | Shared 3.967 s native pair |

The three fresh requests completed sequentially in 0.91, 0.94, and 0.85 seconds. The existing Core worker was ready with queue depth zero before and after. No workers were restarted or provisioned. Raw requests, model output, latency, and health records are in `fresh-core/`.

The generated wave raises a hand but does not completely lower it within the horizon. The look prompt produced subtle upright motion with no clearly established look left. The listen prompt produced both hands rising conversationally rather than quiet listening. Clip labels retain their request identity; `source.observed_action` records the visual result. These are not claimed as perfect semantic completions. Nonloop gestures need an authored exit blend.

The reused source is `review/interaction-quality/after/market-three-s48-fixed-plan/sources/scene-e8470498b43c45f39ed5f0c4b3614fa1/pair-01.npz`. Original source joints remain untouched. Root-centered body poses plus `rootXZ`, one shared floor shift and one shared placement rotation reconstruct every native source joint to floating point precision. The two roles must use unit scale, a common instance yaw, and their distinct root tracks. The source already contains opposing body orientations; independently rotating roles toward each other applies an erroneous second rotation.

`action-cues.json` explains trajectory fields and source start/end facing. Approach/exit headings use common event yaw plus role facing. During pair playback both actor instance yaws use the same common event yaw. Start and end poses are exported for authored transitions. Do not blend the second role to unrotated canonical idle, which faces almost 180 degrees away.

Skeleton contact-sheet review confirms the visible hand raise and conversational gesture. Existing mesh video samples show the reused pair reaching and keeping hands close, including a leaning/bending role. Source wrist distances are under 15 cm for 103 of 120 frames, with a 2.66 cm minimum. This does not establish exact palm/finger contact. Final crowd mesh/browser review belongs to the integrated scene review.

Rebuild without model requests:

```sh
python experiments/crowd_demo_motions.py --flashmob
python -m unittest -v test_crowd_demo_motion_export
```

Explicit fresh generation creates a new source directory and requires a private token file; the exporter never prints credentials:

```sh
python experiments/crowd_demo_motions.py --generate --sources NEW_SOURCE_DIRECTORY --token-file PRIVATE_TOKEN_FILE
```

The atlas output is `review/crowd-demos/assets/{manifest.json,affine.bin,poses.bin,motion-summary.json}`. It retains manifest compatibility with the original GPU crowd renderer. Tests check finite transforms, rig hash and skin weights, offsets and endpoint pose metadata, rejected malformed sources, horizontal root removal, and exact native pair reconstruction.

## Trajectory cue integration

`crowd_demo_cues.apply_motion_cues(trajectory, manifest)` returns a copy with explicit animation fields. The navigation owner calls it before final collision validation and export. It never changes ordinary actor root paths. The six featured actors use the conservative reserved meeting pockets: smooth authored placement from 12–14 seconds, source root tracks from 14–17.9667 seconds, and smooth placement back to their original departure anchors by 20 seconds. Their cumulative distance and speed are recalculated. Per-actor unit scale is required for the whole featured sequence.

Watch/browse schedules receive deterministic short generated cues only while stationary, with settle and departure margins. A moving person keeps their locomotion gait. Gesture events and pair source fields are saved under `motion_cues` and `events`. Full transformed navigation validation is mandatory; the helper removes obsolete pre-cue metrics.

```sh
python crowd_demo_cues.py ORIGINAL_NAVIGATION.json --output VALIDATED_CUED_TRAJECTORY.json
python -m unittest -v test_crowd_demo_cues test_crowd_demo_motion_export
```

The eight focused tests pass. They do not replace full scene collision validation or browser review.


## Reviewed flashmob revision

The published atlas now includes source phrase clips 8 (rhythmic knee-lift steps), 9 (small step/sway), and assembled routine 10. The routine is exactly 25 seconds (751 frames at 30 fps), with neutral bookends and zero horizontal root offset at both boundaries. It combines four reviewed source spans with disclosed authored 0.6-second bridges and a 1.267-second settled exit. The six original source NPZ archives are tracked under `review/group-motion-probe/fresh/dance-three-s73` and `dance-three-s74`. Source model: ARDY Core, seeds 1082 and 74. No new GPU requests were made for this revision.

The city choreography uses 24 synchronized dancers, actor IDs 6–29, in a 6 by 4 formation with 2.6 m spacing and 1.2 m reserved slot radius. Native lateral roots are retained rather than erased. All dancers use unit scale. The same local root offset is applied to every dancer at the same time. Navigation owns arrival, settled slots, and departure; the cue helper refuses unsettled or crowded formations.

Full skin evaluation covers every one of the 28,374 authored rig vertices in all 751 routine frames. The horizontal mesh radius including source root movement is 0.8444 m. The original shoe mesh could dip 9.33 cm below the ground. An explicit authored `groundLiftY` track now supplies a whole-body vertical display offset through **Person[8] only**; original poses, affine transforms, and XZ tracks remain unchanged. The lift is at most 9.53 cm and rate-limited to 0.25 m/s. Ideal 30 fps sole clearance is at least 2 mm. Smooth one-second entry and departure ramps connect the ordinary idle baseline. This is a display correction, not learned motion, foot locking, or physics. Smoothing can leave a few centimeters of sole clearance; exact contact is not claimed.

Original and corrected full mesh contact sheets and bound reports remain under `review/crowd-demos/flashmob-candidate`. Actual 10 Hz trajectory-height interpolation and 0.25-second renderer blend validation is recorded separately there.

Ordinary watch/browse states now use only subtle idle/look cues, globally across all environments. Random waves and conversational hand gestures were removed. The explicit cafe service role and native greeting pairs remain. `--refresh-existing` updates old cue assignments without applying paired root paths a second time. Twelve crossing/station profiles (16, 32, 64, 100, 112, 128 people) were refreshed and revalidated; root fields and paired cues were verified unchanged. Exact hashes are in `ordinary-idle-refresh.json`.

Thirteen focused tests pass, including synchronized dance/root geometry, formation rejection, cue refresh preservation, bounded height ramps, and a full-skinned-mesh regression that detects double application of vertical correction.
