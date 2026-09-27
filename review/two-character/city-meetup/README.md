# Integrated city meetup milestone — 2026-09-26

The main Motion panel now offers **Direct → Two characters**. Pick two cast members, set independent start marks and a meeting point (or an existing scene landmark), preview the routes, then generate a handshake, sparring scene, or a new described interaction. Preview marks can be dragged in the scene. Play, pause, export and the main Save/Open controls work with the native paired archive. Existing single-character G1 generation remains separate.

## Full, uncut studio captures

- `handshake-ui-capture/playback.mp4`: generated through the new UI, 15.7 s. The two people approach, settle, then perform the original reviewed handshake. Accepted for this research demo; approach foot glide and the settling pause remain visible.
- `crossed-sparring-capture/playback.mp4`: 17.7 s. The original straight paths crossed; bounded detours preserve the two roles and give a measured 0.603 m minimum root separation during approach. The subsequent native sparring exchange is readable. It is a non-contact performance, not verified physical hits.
- `standing-hug-capture/playback.mp4`: 13.7 s. Fresh gateway-directed InterGen sample, followed by generated Core approaches. Reviewed as a usable standing greeting/hug and release; arm placement and exact skin contact are not constrained.
- `embrace-capture/playback.mp4`: **rejected** first custom sample. It passed the older numerical checks but dropped one root 0.910 m on entry and did not reliably finish the requested separation. Preserved with all source data. The new 0.30 m entry-height gate rejects this exact output.
- `sparring-live/`: alternate real model run with straight non-crossing approaches. Numerical validation only; not promoted as a visually reviewed result.

Open `watch.html` through the existing local review server at `http://127.0.0.1:24893/city-meetup/watch.html`. Each capture contains a frame-by-frame manifest, exact saved project, first/last frames and a contact sheet. These are complete rendered timelines without cuts, speed changes, or hidden joins. The Studio timeline separately labels **Core approach**, **Authored entry transition**, and **Native paired interaction**.

## Measurements and verification

`results.json` records the actual measured results. Five real scenes used 19 synchronized Core windows and two fresh InterGen samples/two gateway plans on the existing Pod. No new Pod, checkpoint installation or existing worker restart was needed.

| Reviewed scene | Full duration | Generation wall time¹ | Worst final root arrival error | Entry endpoint velocity error |
| --- | ---: | ---: | ---: | ---: |
| Handshake | 15.7 s | 3.77 s | 3.82 cm | 0.104 m/s |
| Rerouted sparring | 17.7 s | 4.90 s | 3.41 cm | 0.158 m/s |
| Fresh standing hug | 13.7 s | 19.68 s | 1.26 cm | 0.062 m/s |

¹ One run per request, includes composition and source archival; custom generation also includes the gateway plan and fresh paired inference. These are observations, not service latency guarantees.

All completed scenes had zero collisions against the **sampled joint/segment scene proxies**, continuous authored floor coverage, and bit-exact native interaction frames. This does not prove mesh clearance, planted feet, finger contact, body-to-body nonpenetration or combat impact. All four captured native projects round-trip exact joints. Library source interactions are fixed reviewed clips, not freshly generated for a changed prompt.

Final checks: **956 Python tests**, **29 client tests**, TypeScript checking and production build passed. The existing large-bundle warning remains. Browser checks covered the new form, route preview, video export after preview (no marker overlays), main native Save/Open including restored request and city, and opening a G1 project while paired mode was active. The stale-marker overwrite and incorrect active-timeline bugs were fixed. `landmark-preview.json` records resolving the actual city streetlamp `city-13` to nearby clear ground (CPU preview only).

## Reproduce

See `docs/PAIRED-DIRECTION.md` for actual-model commands and provider prerequisites. The first UI handshake request is in `handshake-ui-sources/request.json`; the other requests are under `requests/`. Every completed Core27/20fps window retains positions, rotations and its real 330-feature history. Native InterGen source stays native22/30fps. Neither source archive is converted into the other model's history.

Replay the recorded model output through the final checks without inference:

```sh
PYTHONPATH=. .venv/bin/python review/two-character/city-meetup/replay.py
```

This verifies exact accepted output arrays and the rejected embrace's explicit height error; results are in `replay-guards.json`. Earlier failed arch-floor evidence and rejected retargeting experiments are untouched.

Launch the integration studio from this worktree:

```sh
.venv/bin/python director_viewer.py --reference-only --port 2380 \
  --token-path /absolute/path/to/existing/core-token \
  --native-pair-config .runtime/native-pair-provider.json \
  --objects review/scene-integration/live-city.json \
  --native-project review/two-character/city-meetup/handshake-ui-capture/scene.native-pair.stagezero.npz
```

## Release boundary

The integrated orchestration is ready for review as a research demo. It is **not an unlimited or commercially cleared production animation system**. One selected pair acts at a time; extra cast stays idle. Approaches are bounded to 20 seconds and complete takes to 1000 frames. Routes use authored metadata, not visual detection. Bridges are authored and feet are not locked. Custom motion still needs visual review. InterGen's CC BY-NC-SA 4.0 license and the authored display rig's commercial licensing must be resolved before commercial release. No claim of arbitrary grabbing, weapon combat, physical impacts or object transfer is made.
