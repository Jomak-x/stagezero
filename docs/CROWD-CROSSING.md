# Crowd crossing laboratory

This is an isolated first large-crowd experiment. The main Studio UI, native pair protocol and live demo remain unchanged. A recorded AI plan controls route, speed, grouping and gait distributions. A seeded local Python solver produces 120 seconds of crossing trajectories; the browser interpolates those saved trajectories and animates 16, 32, 64 or 100 people with GPU instancing. This version does **not** run its navigation solver in the browser or respond to interactive obstacle changes.

The scene adapts the actual generated city background preserved in `review/prompt-scenes/backgrounds/city.json`. Generated building and streetlamp parts are repositioned around explicitly authored roads, level sidewalks and crossing markings. This is a Shibuya-style scramble layout, not a scan or geographically accurate recreation. Geometry metadata defines navigation; no visual detection is claimed.

Measured results and full playback videos: [milestone review](../review/crowd-crossing/README.md).

## Replay the milestone

Use the repository Python dependencies and `npm ci` in `studio_client` on a fresh checkout. Existing prepared assets and trajectories need no model service, credentials or GPU Pod.

```sh
# Repository root, optional collector for browser results and video:
python experiments/serve_crowd_evidence.py --output review/crowd-crossing/browser-new

# Separate terminal:
cd studio_client
./node_modules/.bin/vite --config crowd.vite.config.mts
```

Open `http://127.0.0.1:24970/crowd.html?count=64&camera=street&evidence=http://127.0.0.1:24973`.
Select a population/camera, then Profile live playback or Record full playback. The latter records the actual canvas in real time; it is not frame-by-frame prerendering. Use `camera=orbit&duration=120&auto=profile` for a moving-camera benchmark. Use a fresh page for each population so startup and payload counters describe that population. Keep the tab foreground and stop unrelated rendering/encoding before accepting timing results.

The evidence collector is loopback-only, serves only its output directory, and accepts POSTs from the exact local harness origin. It never replaces an existing report. The test ports do not replace any existing studio or worker.

## Reproduction and sources

```sh
# Offline trajectory reconstruction from the exact recorded AI plan:
python experiments/reproduce_crowd.py --output .runtime/crowd-reproduction

# Offline motion/background export from retained raw generated sources:
python experiments/build_crowd_assets.py \
  --sources review/crowd-crossing/generation/accepted-sources.json \
  --output .runtime/crowd-rebuilt-assets

# Rendered-motion foot proxies:
python crowd_gait_metrics.py --assets review/crowd-crossing/assets \
  --trajectory review/crowd-crossing/trajectories-64.json \
  --output .runtime/crowd-gait-64.json --sample-fps 60
```

The plan, raw Core NPZs, source requests/seeds, chosen gait windows, affine/pose arrays, generated city provenance and measured generation latency are retained under `review/crowd-crossing`. Seven fresh Core horizons produced two continuous walking styles and an idle plus start/stop sources. The renderer uses the reviewed walks and idle; generated start/stop clips are archived but **not played**. Starts and stops use a labeled deterministic blend, bounded to 1.8 blend units per second. Casual style currently maps to the relaxed walking source.

The native CPU baseline calls the unchanged shipped mesh-skinning and geometry update functions. It is a CPU microbenchmark, not native scene FPS. Actual crowd reports measure rAF, CPU submission, available GPU timers, input hashes, payload, startup and JS heap. The `loadMs` field is local asset/init time after the main JavaScript module begins and before first render; it excludes page/module startup and is not a cold navigation-to-visible measurement. Payload counts fetched scene assets, excluding HTML/JS. JS heap is not total process or GPU memory. The existing native pair protocol limits remain unchanged; this standalone benchmark bypasses that transport. A real WebGL transform-feedback check compares the shared position shader to native CPU skinning; normals remain an approximation.

## Scope and limitations

Navigation uses predictive steering, bounded turning/acceleration commands and residual disc separation. It is not ORCA and has no formal collision guarantee. Collision correction can still create sharp actual root acceleration; root discs are not full-body meshes. The metric reports include pre-correction conflicts, correction distance, remaining overlap, completion, phase occupancy and deadlock proxies. Groups share routes, launch timing and nearby speeds; they can spread around traffic.

Two generated walk styles, one tinted authored Xbot mesh and limited idle variation are not a diverse production character library. Speed follows traveled distance, but foot locking, sole contact and body contact remain unsolved. Reported ankle/toe ground candidates may include low swinging feet. No threshold substitutes for visual review.

There is no featured meetup in this milestone. No geometry LOD or true shadow maps are enabled; lightweight contact blobs provide grounding. GPU animation is exact for the tested affine position formula, while transformed normals approximate the native CPU recomputation. The first milestone remains a separate laboratory until user review approves further integration.
