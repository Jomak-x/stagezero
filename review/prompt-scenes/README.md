# Prompt scenes milestone — 2026-09-26

This is a separately runnable research studio (`prompt_scene_viewer.py`). Main's UI, G1 format, Core format and native pair format remain intact. The new cast archive has its own `stagezero_cast_performance` schema and `.cast.stagezero.npz` extension. Do not call these arbitrary-contact or production-ready interactions.

## What changed

- One prompt determines 1–3 cast members, real scene anchors, ordered actions and automatic duration. No fixed seven-second budget for the entire request. Current bound: 1–4 beats, up to 30 seconds before approaches/transitions, and at most 1,000 display frames; over-budget requests are rejected, never silently truncated.
- ARDY Core generates solo motion and actual travel; InterGen generates each active pair jointly. Three performers change partners in sequence; inactive actors hold an explicitly recorded real source pose. There is no three-body interaction model.
- Native source poses/features are archived before composition. Shared rigid placement preserves paired motion. Labeled, mechanically checked authored transitions connect sources; they are not learned contact transitions or foot locking.
- Automatic starts follow the native pair's roles, avoiding unnecessary side swapping. Close native entries use optional 0.60 m Core arrival spacing and the existing checked entry transition; the 0.55 m navigation gate and all old default behavior remain unchanged. Later pairs choose a bounded clear meeting around stationary actors and walk there with Core. Explicit user anchors remain constraints.
- Idle actors retain the same physical collision proxy. Walking routes add 0.25 m of clearance for arms, with at most three attempts (0.25/0.40/0.55 m routing padding) after a measured idle collision. Each retry preserves native motion and explicit anchors, starts fresh Core history, and retains failed-source evidence. Other errors and mechanical failures are not retried.
- Later approaches begin with the prior actor facing, then smoothly turn toward the route over one second. A prior-to-Core authored transition can use up to one second under the original velocity/anatomy gates; native entry transitions and older callers retain their limits.
- A resident authenticated InterGen worker removes repeated checkpoint loading. One serialized pair-prefetch worker overlaps independent source generation with Core where dependencies allow it; cancellation joins outstanding work and keeps failure/source evidence. The planner has a 16-entry per-instance intent cache keyed by exact prompt, full validated scene and model. Motion is not cached.
- Playback remains local to the browser after its animation payload loads. Prop-aware camera framing considers body and feet. UI upload, scene restoration, capture reservation and atomic cast publication were checked in the actual browser.

## Actual measured runs

All backgrounds come from the existing background pipeline: city from `review/scene-integration/live-city.json`; market/industrial from the original workspace's generated scene-scale evidence. They are preserved under `backgrounds/`.

| Run | Generation wall time | Playback | Qualification |
|---|---:|---:|---|
| Solo wave + celebrate, city | 6.90 s | 8.00 s | Fresh AI plan + real Core |
| Two-person dance, market | 7.63 s | 11.90 s | Fresh plan, Core approach, native pair |
| Explicit ±3 m starts, industrial spar | 11.70 s | 12.60 s | Fresh plan and motion; controlled sparring, not physical impacts |
| Handshake variation 47 | 8.83 s | 7.50 s | Fresh AI plan and motion |
| Handshake variation 48 | 3.81 s | 7.40 s | Same validated intent; fresh motion seed, zero planner time |
| Fresh three-person request through browser UI | 16.65 s | 15.90 s | Fresh AI plan, both native pair clips and all Core; first attempt passed |
| Three-person partner change | 9.33 s | 15.43 s | Reviewed AI plan supplied; both pair clips and all Core motion newly generated; excludes original planning latency |

`repeated-intent/results.json` records the two consecutive full-scene measurements. `experiments/benchmark_prompt_scene.py` reproduces that measurement with a shared planner and different motion seeds. The worker's three warm requests took 2.020, 1.411 and 1.584 seconds including transfer. Handshake parity with the earlier source was bit-exact for joints AND features (`warm-worker/results.json`); warm-start loading took 5.92 seconds. These are request measurements, not universal latency guarantees.

The reviewed handshake's wrists were within 15 cm for 2.47 seconds. This is proximity, not verified palm/finger contact. Source diagnostics are in `source-measurements.json`. The old automatically staged handshake spent 4.7 seconds approaching; replaying its exact paired source with role-aligned starts reduces that to 3.0 seconds (`city-handshake-direct/`).

## Visual evidence and limitations

Videos are complete browser-rendered timelines with one capture per display frame, no cuts or speed changes. Each capture folder has its exact replay archive, a frame manifest, first/last frames and a contact sheet. Labeled Core approaches and authored bridges remain in the videos. The final hand-contact/foot/transition review is recorded in `review-results.json`; geometric acceptance is never treated as proof of good animation.

- `ui-three-recovered-capture/`: complete recovered UI failure, 486 frames. Exact paired sources replayed; real Core regenerated. Reviewed as a functional research result with visible transition foot travel.
- `ui-three-fresh-capture/`: new browser request, 477 frames, all model output fresh. Functional workflow passed; waiting poses look weaker (lean/raised arm), so this is not the preferred display demo.
- `handshake-fast-capture/`: final 7.4-second handshake, fresh seed 48.
- `market-three-final-capture/`: final three-person partner change, fresh pair sources and Core, improved real waiting pose and unobstructed framing.
- `industrial-explicit-spar-capture/`: full industrial approach and spar.
- `market-dance-capture/`: full dance; earlier framing partially hides an actor at entry. Camera correction is separate and does not change motion.
- `solo-city-capture/`: complete solo performance.
- Earlier captures and rejected source archives remain as provenance.

Three-person waiting is stationary, not expressive idle animation. The selected incoming native waiting frame is copied unchanged: in the tested third actor it reduced torso lean from 18.5° to 3.2°. Hand/foot contact is not solved. Cross-model body proportions and some transition foot travel remain visible limitations. No simultaneous three-person hugging/wrestling, arbitrary object transfer, general falls/recoveries, or physical fight simulation is claimed. InterGen remains CC BY-NC-SA 4.0 noncommercial research only.

## Failed cases preserved

The user's dance → trip/fall → help-up → hug request was not fixed merely by assigning more time. The original old workflow squeezed it into one seven-second pair action. The new planner preserves separate actions, but actual trials still rejected unsafe compositions:

- `dance-fall-help-hug/`: Core-to-dance endpoint velocity rejected.
- `dance-fall-help-hug-seed42/`: the solo fall crosses into the waiting performer (0.10 m unpaired root clearance).
- `dance-fall-help-hug-paired/`: context-aware pair source generation still fails a later navigation/entry requirement.
- `market-three-variation48/` and `ui-three-rejected/`: actual Core forearms overlap the waiting actor despite following the root path. Padded routing clears those collisions in `market-three-variation48-recovery/` and `ui-three-recovery/`, but those replays then reject an abrupt prior-to-Core transition. They remain rejected evidence. Final exact-source replays `ui-three-heading-recovery/` and `market-three-heading48-recovery/` pass after gradual initial facing targets (16.2/16.0-second complete scenes). Native sources are replayed; Core is newly generated. These replay times are not fresh end-to-end inference measurements.
- `market-three/`, `market-three-navigation/`, `market-three-standoff/`: preserved anatomy, close-entry and idle-performer-collision failures respectively. The complete three-person result was produced only after actual travel and clear later meeting placement passed.

Failures retain source NPZs and manifests; the live UI keeps the previous complete performance. None was promoted as a successful scene. The existing arch-platform failure evidence is untouched.

## Reproduce on existing services

From this worktree, with the private existing token and provider config (neither is committed):

```sh
.venv/bin/python experiments/benchmark_prompt_scene.py \
  --scene review/prompt-scenes/backgrounds/city.json \
  --config .runtime/prompt-native-provider.json \
  --token /Users/jakob/Desktop/Shellhacks/.runtime/api-token \
  --output .runtime/prompt-scenes/benchmark-new

.venv/bin/python experiments/trial_prompt_scene.py \
  --prompt 'Three people greet each other in turn. Person 1 shakes hands with person 2 and releases. Then person 2 shakes hands with person 3.' \
  --scene review/prompt-scenes/backgrounds/market.json \
  --plan review/prompt-scenes/planning/market-three-replay-plan.json \
  --config .runtime/prompt-native-provider.json \
  --token /Users/jakob/Desktop/Shellhacks/.runtime/api-token \
  --seed 42 --output .runtime/prompt-scenes/three-new

STAGEZERO_CLIENT_BUILD="$PWD/.runtime/local-playback-client" \
.venv/bin/python prompt_scene_viewer.py --port 2382 \
  --background City=review/prompt-scenes/backgrounds/city.json \
  --background Market=review/prompt-scenes/backgrounds/market.json \
  --background Industrial=review/prompt-scenes/backgrounds/industrial.json \
  --token-path /Users/jakob/Desktop/Shellhacks/.runtime/api-token \
  --native-pair-config .runtime/prompt-native-provider.json \
  --project review/prompt-scenes/market-three-fresh/scene.cast.stagezero.npz
```

Use Files and variations → Export video for a complete local browser capture. A replay-only experiment can use `--replay-manifest` instead of `--plan/--config`; its timing explicitly excludes native inference and the interface labels it accordingly. Do not compare that time to fresh generation.

## Verification and resources

1,149 Python tests pass, including localhost transport, cancellation, source preservation, geometry, explicit anchors, archive round-trips, camera occlusion and real Viser upload event shape. Independent read-only review checked publication/capture lock order and later staging. Existing client code is unchanged this milestone.

No new Pod rented. Existing G1/Core/other workers were preserved. The new resident worker uses its own subprocess and loopback port; a private SSH tunnel carries authenticated requests. GPU observation after warmup: 36,268 / 49,140 MiB used, 12,244 MiB free, idle between requests. Shared pod disk was nearly full (~704 MiB free); no weights or dependencies were downloaded. More GPU does not resolve semantic contact/transition limits.

Latest deployment: existing private studio at `http://jakobs-mac-mini.tail5a8376.ts.net:2380/`, with `ui-three-heading-recovery/scene.cast.stagezero.npz` loaded. Refresh the page. RunPod CLI reports existing running costs of $0.53/hr and $0.84/hr ($1.37/hr combined); user authorized a $3/hr total ceiling, approval required above it. No additional Pod was started.
