# Crowd source assets

Run from the repository root:

```sh
/Users/jakob/Desktop/Shellhacks/.venv/bin/python experiments/build_crowd_assets.py \
  --sources review/crowd-crossing/generation/accepted-sources.json \
  --output review/crowd-crossing/assets
```

`manifest.json` contains the authored Xbot mesh parts, exact native22 affine transforms in `affine.bin` (`float32[totalFrames,22,3,4]`), and aligned native22 source poses in `poses.bin` (`float32[totalFrames,22,3]`). Both use Y up and +Z forward. Each clip records its raw Core source archives, hashes, request files, seeds, prompt, selected inclusive source window, and measured source root speed. The browser GPU shader uses the same segment-affine formulation as native playback; comparison against `NativeRigAsset.skin` on the first exported frame differed by at most 0.00000012 m per vertex.

The two accepted walking loops come from four fresh, serial ARDY Core horizons with native history for each second horizon. Relaxed source frames 19–67 produce a 73-frame 30 fps loop at 0.994 m/s with 0.0063 m endpoint RMS per coordinate. Brisk frames 24–56 produce a 49-frame loop at 1.36 m/s with 0.0083 m endpoint RMS per coordinate. Fresh generated idle, start, and stop clips are also packed; current crowd rendering uses the idle/walk clips and an authored velocity blend for starts and stops. The model generated start/stop clips remain available for later transition work. Source job wall times were 0.82–0.96 s each, 6.06 s total for seven jobs; their original `.npz`, full request JSON, individual timings and health snapshots are under `../generation/`.

The initial cached 40-frame approach walks were rejected for crowd looping. Their first/last poses differed by 0.16–0.20 m RMS per coordinate and up to 0.67 m at one foot. `../generation/rejected-cached-loop.json` records source hashes and the discarded packaged manifest hashes.

`crossing_scene.json` adapts the actual generated city snapshot at `review/prompt-scenes/backgrounds/city.json`. The source hash and every reused object/asset/part are recorded in the boxes. We moved 18 generated building, cafe, and tower objects and eight generated streetlamps outside the walkable crossing, retaining generated part colors and proportions. Asphalt, sidewalk strips, waiting pads and orthogonal/diagonal crossing paint are explicitly authored. The walkable proxy is the flat XZ square to ±16 m; the renderer's asphalt top is 1 cm below the actor root plane, sidewalks are at y=0, and paint/pads rise 1–2 mm to avoid coplanar flicker. This is a geometry adaptation of a generated city background, not a newly generated Shibuya asset or image-based walkability estimate.

Current limitations: one authored human mesh is tinted across actors; model body poses provide two accepted walking styles plus idle, so appearance and gait diversity are limited. No foot-contact lock or physical full-body collision is embedded in these assets. Loop seam proximity is a pose measure and does not prove planted feet in browser playback. The selected sources and street-level playback still require full-scene visual acceptance.
