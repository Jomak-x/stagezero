# Validated crowd demo milestone

Three saved 120-second scenes with 112 actors each; separately planned 128-person stress takes. The city was rebuilt after user review to use sidewalks and its two painted zebras. It includes 24 synchronized dancers 40–65 s, 12 spectators, eight shop customers, a café attendant and ongoing street errands. Random ordinary idle waves are removed.

## Films and live viewer

- [Combined 90-second preview](videos/stagezero-three-worlds.mp4)
- [Crossing, full 120 seconds](videos/crossing-112.mp4)
- [Revised city and flashmob, full 120 seconds](videos/city-112.mp4)
- [Station, full 120 seconds](videos/station-112.mp4)
- Live: http://127.0.0.1:24985/demos.html?scene=city&count=112 — use **Courtyard performance** to jump to the dance.

These are actual 30 FPS browser canvas captures, transcoded without speeding up the action. The exact raw film/profile selection and input hashes are in `selected-captures.json`; decoding and media dimensions are checked in `videos/verification.json`. Earlier road-wandering city and empty-center crossing videos are rejected trials, listed in `browser/run-annotations.json`.

## Actual browser measurements

Chrome 154, ANGLE Metal Apple M6, viewport 1200×842, DPR 1. Warm camera/renderer comparisons, 30 s per crossing count and station 128; city 128 ran 60 s. All selected runs were foreground with zero hidden frames. This task’s planner and encoder were stopped for these final comparisons; other user workflows were preserved.

| Scene | Actors | FPS | Frame p95 / max ms | GPU p95 ms | Scene load ms | JS heap peak MB |
|---|---:|---:|---:|---:|---:|---:|
| crossing | 16 | 60.0 | 18.6 / 18.8 | 3.75 | 363 | 58.9 |
| crossing | 32 | 60.0 | 18.6 / 18.8 | 4.75 | 498 | 60.4 |
| crossing | 64 | 60.0 | 18.6 / 18.8 | 5.32 | 541 | 61.9 |
| crossing | 100 | 60.0 | 18.6 / 18.8 | 5.84 | 565 | 65.7 |
| crossing | 128 | 60.0 | 18.6 / 18.8 | 6.19 | 619 | 101.9 |
| city | 128 | 60.0 | 18.6 / 18.8 | 6.13 | 291 | 111.7 |
| station | 128 | 60.0 | 18.6 / 18.8 | 6.85 | 322 | 68.7 |

These are warmed playback rates, not a guarantee of hitch-free first navigation. Two fresh-page comparison trials at 16/32 actors dropped to 48.2/48.3 FPS with CPU render-submission stalls and frame maxima 466.7/401.8 ms; same-context repeats returned to 60 FPS. Their GPU p95 remained 3.74/4.74 ms. Cause is not proven; the outliers are retained. The count matrix uses a shared warmed WebGL context after the repeats. Scene load covers data fetch/parse, geometry initialization and first render; initial module time is recorded separately. For same-page count switches the atlas is already resident, and payload bytes/input hashes accumulate across that page session. JS heap is not process or GPU memory.

The original native CPU deformation bottleneck was measured before the GPU renderer was introduced: see [native audit](../crowd-crossing/README.md). This viewer shares native affine poses across instanced meshes. Shader position parity passed at roughly 2.0e-7 m maximum error on 255,366 scalar comparisons; normals remain approximate.

### Full recorded takes

| Film | FPS | Frame p95 / max ms | Heap peak MB |
|---|---:|---:|---:|
| crossing 112 | 59.7 | 18.6 / 233.3 | 151.0 |
| station 112 | 59.8 | 18.6 / 67.6 | 162.3 |
| city 112 | 60.0 | 18.5 / 18.8 | 196.3 |

Recording includes the video encoder, and parts of these film runs overlapped offline planning. The city film itself had no frame over 18.8 ms; the crossing and station films retain their occasional stalls.

## Behavior checks and candid limits

- All 18 archives pass sampled static/root-disc checks, sampled and swept pair-disc checks, walkable bounds, root-jump and unscheduled-stall checks. Minimum swept separation is 0.65 m. This does not certify full-body/hand collision.
- City 112: zero unpainted-road samples, 53 zebra users, 35 completed opposite-side crossings, 24 dancers settled by 39 s, zero outsider intrusion into the dance area; every dancer moves at least 9.9 m away afterward. City 128: 65 zebra users and 45 completed crossings.
- Crossing 112: 109 people at the central peak; all 112 reach an opposite corner. Road occupancy is zero during WAIT.
- Planned queues can be long: crossing 112 up to 42 s, city 112/128 up to 48/60 s. Café staff intentionally stay at work and spectators pause to watch. Zero unscheduled deadlocks does not mean every actor continually travels.
- General root paths still have abrupt turns and velocity changes, with peak acceleration proxy 36.77 m/s². Foot sliding remains: low-foot endpoint bout drift p95 is about 0.50 m crossing, 0.51 m city and 0.48 m station. These proxies can include low swinging toes and are not physical sole contacts.
- The curated dance combines reviewed native step/sway phrases, with original lateral roots and disclosed authored transitions. A rate-limited whole-body vertical correction reduces measured skinned-shoe penetration to 9.11 mm worst case at actual playback sampling/blending. The library is reused; these are not 24 independent neural dance generations.
- The opt-in terrain kiosk trial generated native motion but failed display clearance/rotation checks; the station stairs are unsupported. Those rejected trials are excluded; see [terrain audit](terrain/README.md).
- Three fresh Core gesture generations took 0.85–0.94 s each; six kiosk horizons took 7.62 s. No new Pods were rented; the observed shared running cost was $1.37/hour. Playback needs no generation service.

## Reproduction and validation

See [README](README.md), [navigation/rebuild instructions](NAVIGATION.md), [motion provenance](motions/README.md), [visual review](visual-review.md), `navigation-metrics.json`, `navigation-archive-hashes.json`, `browser-summary.json` and `source-evidence.json`. All source, curated motion and deterministic presets are saved in the isolated checkout; the original dirty UI/workers were preserved. 29 motion/gait/terrain tests plus 9 navigation tests passed; TypeScript and the production demo build passed. This is the review milestone, not a production or physical-contact certification.
