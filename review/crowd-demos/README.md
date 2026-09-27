# StageZero: a day in motion

Three saved 120-second scenes, each with 112 colored Xbot actors and a separately planned 128-actor stress take. These are precomputed demonstrations: playback needs no generation service or rented GPU. Every actor has a saved intent, repeated destination/task schedule, motion cues and a stable ID. The revised city uses sidewalks and marked crossings, with 24 people gathering for a synchronized courtyard performance before dispersing. Ordinary idle pauses use subtle motion; conspicuous gestures belong to actual encounters or café service. The city has an intentionally stationary café attendant who gestures and serves; stationary work is distinguished from a navigation deadlock.

See [measured results and films](RESULTS.md) for the validated review milestone.

## Open the demos

### From the normal Studio app

After merging PR #41, build the client (`cd studio_client && npm ci && npm run build`) and start/restart Studio using your usual launch command. The server caches assets, so refresh alone is insufficient after replacing a running build. In the main controls panel, choose **Crowd demos**. It opens a separate tab on the same app server, preserving your working Studio tab. No GPU worker or separate demo server is needed. You can also open `/demos.html?scene=city&count=112` on that server directly.

Choose **Crossing**, **City lives**, or **Last train**, then select **112** or **128** people. Drag to orbit, scroll to zoom, use the timeline to inspect any moment, and select a person followed by **Follow actor**. Camera buttons choose fixed views; **Director camera** resumes the movie sequence. **Courtyard performance** jumps to the city dance. **Back to Studio** opens the Studio entry in the current tab; your original Studio tab stays open.

The normal build includes all18 saved populations and the animation atlas under `build/demo-data/`. Only the selected take is fetched at runtime. On ordinary builds, **Save still**, **Record full demo**, and **Profile** produce explicit download links below the controls, with no server write API. The separate local capture server keeps its existing evidence-save workflow.

### Standalone development viewer

From the isolated checkout, run:

```sh
cd studio_client
./node_modules/.bin/vite --config demos.vite.config.mts
```

Open http://127.0.0.1:24985/demos.html. Choose Crossing, City lives or Last train. The director switches between six cameras and the featured native greeting. Pause and scrub, choose a person to inspect their current task, use Follow actor, or drag the view. In City lives, Courtyard performance jumps directly to the dance. Replay restarts the saved take after its deliberate 120-second end. It does not teleport actors in a seamless loop.

The 16/32/64/100 versions are independently checked subsets of the 112-person take, for rendering comparison. They are not separately optimized crowd simulations. The 128-person version is separately planned. Query parameters `scene=city&count=112`, `camera=street|overhead`, `auto=profile|record` and `duration=120` make browser measurements reproducible. Recording captures the actual WebGL canvas at 30 FPS, excluding the HTML controls, and saves its JSON profile plus WebM into `browser/`. Use the 120-second default for full-scene evidence.

## What is generated and what is authored

The environments combine authored navigable 3D geometry with actual generated image assets, retained from the background milestone. They are fictional Tokyo-inspired stages, not photogrammetric reconstructions. The crowd uses reviewed generated native locomotion, three freshly generated Core gesture clips and an existing jointly generated InterGen greeting. See [motion provenance](motions/README.md), its raw NPZ/request evidence, and the atlas source hashes. The look prompt only produced a subtle idle; the listen prompt produced a conversational hand gesture. Neither is mislabeled as a perfect prompt completion in the source review.

Routes, role descriptions, task schedules, meeting placement and cameras are authored offline. A bounded Neon GPT-6 Astra architecture review informed the design; it is not a live per-person AI controller. All six camera views are warmed before the take starts, so first-use work is included in load time. The flashmob preserves generated step/sway phrases, including their native lateral root movement, assembled with disclosed authored transitions. A measured, rate-limited vertical display correction reduces courtyard shoe penetration to a measured 9.11 mm worst case at the actual playback sampling rate; it is not a claim of learned physical contact. The browser uses two instanced body meshes and shared GPU affine pose textures. It does not run the native CPU deformation loop for each actor. The six featured actors retain unit scale and jointly generated relative roots so the pair motion is not independently rotated or rescaled.

## Acceptance scope and limits

Navigation checks cover all cached frames plus continuous linear swept-disc distances between them, conservative authored obstacle footprints, walkable bounds, root jumps and unscheduled stalls. Passing these checks does not certify full skinned-body or hand collision. Bodies can extend beyond their 0.28 m root discs. Shop customers pause at their destinations; they do not physically grasp or exchange merchandise.

Crossing movement comes in signal phases rather than constant occupancy. The revised 112-person take peaks at 109 people in the central 25.52 m square and clears the road during WAIT. Some actors miss a crossing window: maximum continuous stationary time is 50.5 s, with a 42 s explicit signal queue. Idle motion continues during those waits; zero deadlocks does not imply every pause is brief.

Root paths are continuous but piecewise linear: sharp route vertices and dwell starts/stops have discontinuous velocity. Heading easing and locomotion blending reduce visual abruptness but do not establish physical acceleration or planted feet. Foot-proxy drift is reported separately; these endpoint proxies are not mesh soles or contact forces. Two walking styles and a small reviewed action library are reused with different phases and colors. The dance is a curated shared routine, not 24 independent neural generations; animation diversity remains limited.

The opt-in terrain trial is deliberately separate. It produced a real 240-frame native kiosk detour, but the display solver failed its contact/rotation checks and a limb collided with the kiosk. The exact station staircase also has unsupported geometry. Those rejected motions are excluded from the accepted scenes. Read the [terrain audit](terrain/README.md), including the CPU diagnostic that separated a false foot-support lookup from a real clearance problem.

No additional Pods were rented. The read-only inventory showed the existing two running Pods at $1.37/hour combined, below the user’s $4/hour account cap. Existing workers and the live application were not restarted or replaced. Three gesture generations took 0.85–0.94 seconds apiece; the bounded six-horizon kiosk generation took 7.62 seconds. Playback itself consumes no model generation.

## Reproducible source

- `experiments/build_crowd_demos.py`, `crowd_demo_navigation.py`, `crowd_demo_layout.py`: deterministic offline planner and authored geometry proxies.
- `crowd_demo_cues.py`: native pair roots and reviewed stationary action cues, followed by independent validation.
- `experiments/crowd_demo_motions.py`, `crowd_demo_motion_export.py`: archived native sources to affine atlas.
- `studio_client/src/demos/`: isolated GPU viewer, actor inspector, director and measured canvas recording.
- `crowd_demo_gait_metrics.py`: rendered native22 foot proxy analysis matching viewer cue interpolation.
- `navigation-metrics.json`: all saved populations, including failures at zero when validated.
- `browser/`: actual browser profiles, input SHA-256 values, raw captures and selected stills.

Browser heap measurements exclude GPU memory and total browser process memory. Profiles record the exact renderer, viewport, capture state, load scope and hidden-frame count. Recorded playback includes video encoder overhead; a short stress run does not prove a whole two-minute take stays at that rate. Do not compare these rich-scene results directly with the older empty-stage baseline without accounting for geometry, viewport and camera.
