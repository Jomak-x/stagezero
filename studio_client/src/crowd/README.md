# Isolated crowd browser harness

This additive experiment leaves the studio UI and `NativePairPlayback.ts` unchanged.

Run from `studio_client`: `./node_modules/.bin/vite --config crowd.vite.config.mts`.
Open `http://127.0.0.1:24970/crowd.html`. The development config exposes the isolated checkout's `review/crowd-crossing` assets. A static production build requires serving those assets alongside the bundle; they are not copied into the build.

## Native baseline

Use `?auto=native&evidence=http://127.0.0.1:24973`. The baseline invokes the shipped `skinNativePairFrame` and `NativePairGeometry.update` with actual exported Xbot bindings and generated affine motion. It measures 16,32,64,100 independently allocated actors; 5 warmup iterations precede 30 recorded iterations. Both skin-only cost and full endpoint-normal/interpolation/bounding-sphere update are measured. Timer yields occur outside measured sections. This is a CPU microbenchmark, **not a native browser FPS result**; renderer submission and GPU upload are excluded.

## Live measurement and video

Example: `?count=64&camera=street&duration=120&auto=profile&capture=1&evidence=http://127.0.0.1:24973`.

- Populations:16,32,64,100. Cameras:`street`,`overhead`,`orbit`.
- Real `requestAnimationFrame` wall-clock deltas drive timing and scene advancement. Saved navigation is interpolated. Playback clamps at the final pose and pauses; only Restart scene begins another run, preventing an end-of-capture reset jump. Measurement continues for its requested wall-clock duration while an ended scene holds its final pose.
- `capture=1` records the same live canvas with MediaRecorder requesting30fps. Actual delivery depends on scheduling/encoding. This is not deterministic frame export.
- Reports include rAF median/p95/p99, sustained fps, animation CPU, render submission CPU, optional disjoint GPU timer, visibility, resolution/DPR, user agent/GPU, load time, fetched bytes, SHA256 of all loaded input bytes, JS heap (when exposed), draw calls/triangles.
- Submission CPU is not GPU time; JS heap is not total process/GPU memory. Visibility cannot reliably detect OS screen locking. Require unlocked foreground playback and visual review.
- `/report?name=...` receives JSON; `/video?name=...webm` receives video. Without `evidence`, files download.
- Use a fresh URL per population: loadMs describes initial page load and payloadBytes counts all fetches on that page. Controls are disabled during measurement.

## Representation and limits

A shared float32 texture stores exact native affine rows `[frame,22 bones,3 rows,4 columns]`. Geometry keeps all four per-vertex bind positions/influences. Each part uses one instanced draw; vertex shaders sample transforms, interpolate source frames, blend idle/walk and place actors. No vertices or normals deform on CPU during playback.

Ground is XZ, Y up, heading0 faces+Z. Phase advances with root distance divided by clip travel, adjusted for instance XZ scale. Height, width, phase, color, route and source clip vary deterministically. Walk candidates have type `walk` or `walk` in the id; brisk agents use a clip containing `brisk`, relaxed/casual agents a clip containing `relaxed`. Fallbacks are the second/first walking clip. Generated start/stop clips are retained as source evidence but not selected as looping locomotion.

Starts/stops use a procedural smoothstep over speeds0.025–0.42m/s, rate-limited to1.8 blend units/second at saved navigation samples and interpolated between samples. This is deterministic when seeking and suppresses collision-speed spikes; it is not contact IK or a learned transition. Idle poses sample source frames without subframe interpolation. Loop seams and foot sliding require independent metrics and whole-scene review.

Normals use weighted affine-transformed rest normals, not deformed triangle normals. Radial translucent root contact blobs provide grounding and are **not true shadows**. No shadow maps or geometry LOD are enabled. Pixel ratio is capped1.5. Background shapes are instanced by color/shape; boxes honor `yawRadians`, spheres/cylinders use coarse geometry. Metadata is not visual detection.

Every live-profile page checks real GPU positions using WebGL2 transform feedback against native CPU deformation: all vertices, three source/blend cases, tolerance0.00001m. The exact shared shader position source is tested. Approximate normals are explicitly excluded from equality claims.
