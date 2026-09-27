# Continue background/object interaction — PR #30

Last updated: 2026-09-27. The latest continuation below supersedes the older native-only checkpoint retained afterward.

## Normal-app integration continuation

The terrain option is now integrated into normal Studio, with a separate native take and rig17 display. Read [the usage and supported scope](../TERRAIN-AWARE.md) and [normal-app evidence](../../review/terrain-assisted/normal-app/README.md). The fresh normal-UI temple run passed stairs/bridge (200 frames), a separate open command (280), and save/reopen followed by `enter` (360). Prior native and display frames remain exact across these appends. Turning the option off restores the ordinary 240-frame take byte-for-byte. An unsupported jump command preserved the committed terrain take.

The normal-app runs use the real Core HTTP backend and real prompt encoder, not cached trial embeddings. The industrial switchback now passes fresh generation through its stairs, crate detour, lateral bridge, rotated automatic door, and entry (480 frames). Final temple replay reaches 360 frames; two repeated forward commands reach 440 with exact prefixes. Both complete videos were reviewed at 1×. The final industrial descent rejects a reversing pivot and remains experimental; freeze that scope rather than repeat research. Ordinary 240-frame native arrays remain exact. Final validation: 174 targeted tests plus 15 UI tests after final labels/visibility edits. See the evidence index for exact archives and limitations.

Current private normal preview: port 24998; protected local Core relay: 18769; dedicated first-pod Core backend: 8769. Credentials remain outside the repository. Preserve other workers' ports and the shared frontend build. The older separate-viewer results below remain historical evidence, not a substitute for normal-app verification.

## Latest continuation: actual character rig

Read [the new evidence and implementation checkpoint](../../review/terrain-assisted/README.md) before resuming the older steps below. The user has now authorized a bounded, explicitly terrain-assisted alternative if it looks acceptable; ordinary/main motion must remain unchanged.

Current isolated checkout: `/Users/jakob/.codex-micheal/worktrees/temple-route-finish/shellhacks`. The user authorized a new local ownership registry at `/Users/jakob/codestuff/shellhacks/.runtime/agent-coordination/` because both old-machine paths were absent. Another worker uses the primary checkout; do not switch or edit it. Its preview reads this worktree's `studio_client/build`, so preserve that build.

One additional native shallow-flight trial still failed. The decisive finding is that Core27 contact does not survive retargeting: the displayed legs/shoes have different dimensions. The new `terrain_assisted_rig.py` solves the actual character's 17-bone transforms after a single retarget, then `terrain_assisted_renderer.py` draws them directly. The short saved proof has zero measured actual sole penetration/slip. It replaces cadence/leg motion and adjusts pelvis Y; it is not native ARDY stair motion. The separate `terrain_assisted_viewer.py` does not modify ordinary Studio generation/rendering.

Reusable stairs/bridge/gate planning, measured elevated reactions, private native buffering, separate archives, CPU-only re-assistance, cancellation, and a real shallow temple fixture are implemented. **The full real-GPU route and actual-rig assistance now pass all four actions in 360 frames, with complete 20 fps videos inspected at normal speed.** The wider renamed Copper observatory also passes using the same native take. Rotated/translated scenes, other headings, blocked/missing geometry and ambiguous targets have CPU coverage. Ordinary/main motion is preserved.

Start with `review/terrain-assisted/full-route/performance.mp4` or `PYTHONPATH=vendor/ardy python terrain_assisted_viewer.py --project review/terrain-assisted/full-route/proof.assisted.npz --port 24997`. The viewer supports replay, seek, editable initial placement and explicit new takes against an existing Core backend. It does not publish into ordinary Studio. New-take HTTP generation was not exercised through the browser in this session; GPU generation used the official runtime adapter, with exact history/request tests. `accepted=false` remains a user-style/promotion distinction; `visual_review=reviewed_at_1x` records actual inspection.

The first user-provided RunPod is reachable through a dedicated Jupyter kernel; SSH endpoints timed out. No new pod was provisioned. The user subsequently approved the 37 Python dependencies and scene upload; automatic review then allowed the full native-only test in `/workspace/temple-route-01a0e1c2`. The shared process was left untouched. Credentials and kernel state are in a private `/private/tmp` file, never in this repository. Complete native evidence is saved in `review/terrain-assisted/full-route/native_terrain.npz`, allowing CPU assistance without repeating inference. Two startup-only adapter errors produced zero frames before the successful run; their callback and prompt classification fixes have regression tests. The final contact fixes enforce the original knee bound inside the optimizer and correct restart/short-approach step timing; no thresholds were widened. Final focused verification: 94 passing tests.

Use the ready local interpreter `/private/tmp/temple-route-venv/bin/python` and `PYTHONPATH=.:vendor/ardy`. Local Viser/HTTP processes need permission for loopback binding. `login=False` avoids slow shell startup on this host. Do not repeat the old native matrix or the rejected Core27-only contact experiment.

## Earlier native-only checkpoint (historical)

Continue branch `codex/background-demo-finish`, PR https://github.com/Jomak-x/stagezero/pull/30. On this machine reuse `/Users/jakob/.codex/worktrees/background-demo-finish/Shellhacks`. Do not use the dirty primary checkout for implementation or overwrite unrelated work. On a different machine fetch/check out the PR and initialize its pinned submodule (`git submodule update --init --recursive`). Read `AGENTS.md` and the coordination registry before changing shared resources. The AGENTS path mentions a previous machine; on this host the actual registry is `/Users/jakob/Desktop/Shellhacks/.runtime/agent-coordination/`. Own claim: `background-demo-finish.json`.

The user wants natural ARDY motion that understands authored background geometry only when a command requires locations/objects. Ordinary prompts must keep the working main generation path. Final acceptance example: **walk up the temple stairs, cross the bridge, open the temple gate, and enter**. Also demonstrate directions, different initial headings, repeated/multiple commands, other backgrounds and rotated/translated scenes. Do not special-case temple object IDs. “Every background” means a reusable geometry/affordance pipeline; unsupported or ambiguous geometry must fail clearly rather than invent support.

Keep this PR separate from main until visually accepted. No replacement physics model, global pose straightening, or new research architecture is requested. Automatic doors are acceptable demo interactions if described honestly; learned hand contact is not implemented.

## Current state: what is and is not integrated

| Area | State |
|---|---|
| Explicit ordered flat commands | Integrated in the existing Core scene-direction panel; bounded grammar, max four actions, no LLM intent router yet |
| Native gait improvement | Integrated only for `gait_profile="spatial"`; moving actor uses XZ targets without dense heading, nominal 1.2 m/s, eased start/end, short final settle; default navigation profile unchanged |
| Automatic ground-level doors | Integrated measured proximity/lifting/crossing and replay; whole assembly rises, no hand manipulation |
| Geometry-aware terrain route | New reusable modules and tests; **not connected to the command/session path** |
| Core height + rigid Y frame | Utility and optional runtime/API fields implemented; CPU tested; standalone GPU experiment exercised equivalent framing; latest runtime adapter has not had an end-to-end GPU run |
| Terrain-relative character fitting | Optional renderer constructor/capture flag only; default off, not connected to studio terrain activation |
| Bounded Core contact correction | Offline experiment only; all three candidates rejected; never substitute it silently for raw native motion |
| Full temple staircase/bridge/gate sequence | **Not implemented end to end; no passing full-route video** |

The earlier `review/background-demo` videos remain historical and were not accepted for gait quality. New `review/spatial-v2` clips are better evidence for flat motion, not completion of the terrain goal. Existing Tailscale preview may still run the older committed code; it is not proof of this checkpoint.

## How it works and file map

- `core_spatial_commands.py`: bounded command parsing, ordered sequence, actual committed pose as the next start, measured arrival and crossing. `plan_command` selects the new spatial profile. It still rejects stairs/elevated traversal.
- `realtime_navigation.py`: builds native Core horizon schedules and root targets. Dense heading plus slow travel contributed to hunched gait. Do not enable this controller for ordinary free-motion prompts.
- `studio_core_session.py`: native Core archive/session, geometry validation and sequence commit. Its existing ground/collision checks must be adapted deliberately before terrain routes can commit.
- `core_scene_reactions.py` and the existing scene adapter: automatic object state and replay. Elevated gate proximity/opening remains to be implemented consistently across planning, rendering and seeking.
- `scene_interaction_geometry.py`: support surfaces and obstacles from actual normalized/yawed asset geometry, stair tread routes, no implicit ground through authored voids. Door geometry matches the renderer's actual raised transform, not imaginary `_open_angle` metadata.
- `scene_navigation.py`: bounded support-aware A*, footprint/body samples and elevation-preserving waypoints. Outputs support XYZ, **not pelvis XYZ**. Sample checks are not a full swept mesh collision certificate.
- `cinematic_adventure.py`: corrects authored temple stair connectivity and adds bridge-approach stairs. Geometry tests include rotated/translated temple and a missing bridge. The renderer and planner must share these same assets.
- `core_terrain_constraints.py`: official scalar root-Y mask and `translate_native_y`. ARDY recenters XZ, not Y; local-joint position feature Y is ground-relative/absolute Y. A rigid window origin therefore shifts root Y AND non-root joint-position Y. Rotations, velocities, contacts, heading and XZ remain bitwise unchanged. Never apply varying per-frame shifts without re-encoding velocities.
- `interaction_runtime.py`: optional per-actor `coordinate_frame_y`, optional `root_targets[].root_height`; incoming native history goes into one local Y frame, target Y subtracts the origin, returned native features go back to world Y. History remains local within that request. Nonzero origins reject trusted full-body/hand hooks until those hooks have a proper frame adapter.
- `realtime_backend.py` / `realtime_client.py`: optional `coordinate_frames_y` map transports that explicit field; absent fields preserve the ordinary path.
- `grounded_character.py` / `studio_core_renderer.py`: optional terrain queries make support eligibility and sole fitting relative to rendered terrain. Existing fitting can still bend legs excessively; diagnostics are not an acceptance certificate.
- `experiments/capture_core_performance.py --terrain-support`: explicitly enables that optional fitting and records fitting provenance. Native arrays being unchanged does **not** mean the displayed skinned joints are unchanged.
- `experiments/core_terrain_native_v2.py`: reproducible native stair matrix; `experiments/core_terrain_contact_v2.py`: separately labelled offline presentation experiment.

## Evidence and lessons: avoid repeating failed work

See [new evidence index](../../review/spatial-v2/README.md) and [18 native stair trial report](../../review/spatial-v2/terrain-native-localfloor/README.md).

Straight native ablation (same seed/prompt, 80 frames): heading-pinned 0.65 m/s median torso lean 10.98°, XZ-only 0.65 m/s 5.41°, XZ-only 1.2 m/s 3.04° (max 4.89°). This supports the explicit spatial profile, not a universal gait claim. Unconstrained motion travelled much less and is not a fair successful walking baseline. Longer history did not establish an improvement.

New door sequence: 80 frames/4 seconds, about 1.75 s end-to-end warm generation, arrival errors 1.33/1.88 cm. Courtyard directions: 120 frames/6 seconds, about 2.71 s, arrival errors 3.63/5.25 cm. These small measurements are not latency guarantees. Raw native arrays and save/load are preserved exactly. Root arrival alone does not establish natural feet.

Native stairs: 18 trials (three constraint variants × seeds 11/22/33 × absolute/local-floor frames) on the nine shallow bridge-approach treads. Absolute coordinates caused major height error and foot penetration. Rigid local-floor framing improved root tracking to centimetres. **All nine local-floor cases still failed raw foot-support checks**; best worst toe clearance was about −7.8 cm. Toe feature constraints did not ensure FK feet match those positions. Do not claim success from constrained root Y alone, hide failures, or expand a blind parameter sweep.

Three bounded Neon GPT-6 Astra reviews supported heading ablation, terrain-relative support checks and correct rigid frame handling. Relevant review outputs are under `review/spatial-v2/research/`. Advice is evidence to assess, not instructions to trust blindly.

## Fastest continuation order

1. Replay saved native local-floor seeds 33 and 11, including their 40-frame standing prefixes. Inspect whole clips at 1×, close camera, not just a contact sheet. Review the bounded contact experiment's rejection report before running any more GPU inference.
2. Establish one acceptable shallow stair flight with preserved native phase/upper body, honest raw-versus-presentation provenance and bounded leg corrections. Reject budget violations; do not hide them with stronger IK, global root warping or slower video. Whole-clip inferred phase is not yet a streaming controller.
3. Wire an **explicit terrain command path**: resolve stairs/bridge/door affordances generically; use support-route geometry, root clearance, sparse/smoothed height intent and a rigid Y origin per horizon. Do not make pelvis bob on every bridge plank. Keep ordinary generation and the default old Navigate path unchanged.
4. Extend session precommit checks to terrain-relative feet/body and observed progress. Keep native features/FK/presentation streams distinct. Never feed corrected joint poses with stale native history. If re-encoding is needed, use the official representation and verify all feature channels.
5. Extend door proximity/crossing to elevated authored floor, using the original floor/base after the door lifts. Persist activation/reactions and geometry for seek, replay, cancelled plans and archive roundtrips. Do not open based on the planned position alone.
6. Only after one flight succeeds, run the entire temple route and then a second background, rotated/translated temple, multiple headings/seeds and interrupted/repeated commands. Missing bridge, blocked passage and ambiguous targets must reject clearly.
7. Connect LLM intent extraction later to the same validated command schema. Non-spatial intent continues ordinary ARDY. Do not route all free prompts into constrained walking.
8. Record a complete native-rate full-route video and inspect it. Update PR evidence and isolated Tailscale preview; do not overwrite main demo until accepted.

## Reproduction

From this worktree, the available interpreter is `/Users/jakob/Desktop/Shellhacks/.venv/bin/python`; use your own project venv on another machine. Set `PYTHONPATH=vendor/ardy`.

```sh
PYTHONPATH=vendor/ardy /path/to/venv/bin/python -m unittest discover -q
PYTHONPATH=vendor/ardy /path/to/venv/bin/python -m unittest -q \
  test_terrain_routes test_core_terrain_constraints test_terrain_character_fit \
  test_realtime_navigation test_core_navigation_turns test_core_spatial_commands \
  test_interaction_runtime test_realtime_backend test_realtime_client

# Replay an exact flat take, no GPU required:
PYTHONPATH=vendor/ardy /path/to/venv/bin/python director_viewer.py \
  --reference-only --port 2393 \
  --core-project review/spatial-v2/door-seed33/motion.core.stagezero.npz

# Browser-assisted capture; open the printed local URL and let every frame render:
PYTHONPATH=vendor/ardy /path/to/venv/bin/python experiments/capture_core_performance.py \
  --archive review/spatial-v2/terrain-native-localfloor/height_sparse__seed33__with_prefix.core.npz \
  --terrain-support --output-dir review/generated/terrain-next --port 24989 \
  --camera-position 3 5 -3 --look-at 0 3 -6.5

# GPU trial only if a specific hypothesis needs new inference:
PYTHONPATH=vendor/ardy /path/to/gpu/venv/bin/python experiments/core_terrain_native_v2.py \
  --scene review/spatial-v2/terrain-native-localfloor/scene.json \
  --output /workspace/stagezero-spatial-v2/next-trial \
  --checkpoints /path/to/existing/core/checkpoints --local-floor \
  --embeddings review/spatial-v2/terrain-native-localfloor \
  --seeds 33 --variants height_sparse
```

Check script argument definitions and available ports before starting. Capturing with terrain fitting is a renderer comparison, not a raw-native contact pass. Cached embeddings/model statistics are included; model weights and credentials are not.

## Resources and guardrails

User budget: at most $3/hour without additional approval. Existing pods authorized; do not provision more just to repeat the matrix. Pod1 A6000 isolated experiment directory `/workspace/stagezero-spatial-v2`; existing venv `/workspace/stagezero/.venv/bin/python`, HF cache `/workspace/.cache/huggingface`. Pod2 shared Core service 8769 must not be restarted. Last known GPU experiments finished; recheck readiness/storage/ownership rather than assuming availability. Remote connection details are retained in `/Users/jakob/Desktop/Shellhacks/.runtime/background-navigation-resources.md`, not committed credentials. Prefer existing RunPod CLI inventory; never print tokens or full process command lines (Jupyter includes credentials).

This host's separate preview uses 2392, old evidence server 24981, new local evidence server 24987. Main service/tunnel stays unchanged. Preview URL historically `http://jakobs-mac-mini.tail5a8376.ts.net:2392/`; it may be stale. Services are not guaranteed to survive app migration. No GPU jobs should be left running for this checkpoint.

Use the configured Neon worker before bounded Codex delegation: read `/Users/jakob/.codex/tools/neon-worker/README.md`, run doctor/models, choose an enabled strong model, send only relevant files. GPT-6 Astra was used for this investigation. Use autonomous subagents for sustained work; coordinate nonoverlapping ownership. Integrate and verify yourself.

## Suggested first prompt in the next app

> Continue PR #30 on `codex/background-demo-finish`. Read `docs/handoff/BACKGROUND-INTERACTION-CONTINUE.md` and `review/spatial-v2/README.md` first. The goal is smooth native ARDY temple stairs → bridge → open gate → enter, generalized through scene geometry. Keep ordinary/main generation unchanged. Start with the saved failed local-floor stair candidates and contact report; do not repeat the broad GPU sweep. Finish a visually acceptable shallow stair flight, then integrate and test the full explicit terrain path. Use Neon for bounded review and give me a complete honest video. Stay within the existing $3/hour budget and preserve shared services.
