# First crowd milestone — review checkpoint

A 64-person Shibuya-inspired scramble crossing now plays in a separate browser laboratory, with 16/32/100-person configurations. Generated city parts, fresh generated walks and a recorded AI plan drive the scene. The rendering/scale milestone is validated on this machine; believable close-up foot contact and smoother dense avoidance still need work. Stop here for user review before Studio integration or further motion work.

## Actual full playback

- [Street-level, 64 people, full 120 seconds](videos/street-64.mp4)
- [Overhead, 64 people, full 120 seconds](videos/overhead-64.mp4)
- [Street contact sheet](videos/street-contact-sheet.jpg) · [overhead contact sheet](videos/overhead-contact-sheet.jpg)

These are real-time browser canvas recordings, transcoded from VP9 WebM to H.264 MP4 without retiming. They are visual evidence, excluded from the controlled performance table because unrelated work overlapped recording. Raw local captures and hashes are retained; see `videos/provenance.json`. Preliminary and rejected runs are labeled in [run annotations](browser/run-annotations.json).

## Browser measurements

Mac mini, Apple M6, 12 CPU cores, 32 GB RAM; Chrome154 / ANGLE Metal. Viewport1200×808, DPR1. One120-second foreground run per population, moving camera, same geometry/quality, no recording or competing generation/encoding/tests;1.5-second warmup. These are real browser timings, not prerender throughput.

| People | FPS | Frame p95 / max (ms) | GPU p95 (ms) | Asset/init (ms) | Assets (MB) | JS heap max (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 60.00 | 18.5 / 18.8 | 1.49 | 244.1 | 13.26 | 59.90 |
| 32 | 59.99 | 18.4 / 34.9 | 2.33 | 269.7 | 15.88 | 125.58 |
| 64 | 60.00 | 18.5 / 18.8 | 3.77 | 343.2 | 21.01 | 80.67 |
| 100 | 60.00 | 18.6 / 18.8 | 5.18 | 545.0 | 26.93 | 173.42 |

Asset/init starts after the main JS module begins and ends before first render; it is **not cold navigation-to-visible time**. Payload excludes HTML/JS. Heap is browser-reported JS allocation, not total process/GPU memory; GC accounts for nonmonotonic peaks. No hidden frames in accepted runs. Full samples, hashes, CPU submission timings, GPU timers and device data are in [browser reports](browser/) and [review-results.json](review-results.json).

The initial native CPU audit established the bottleneck before implementation. A controlled repeat measured median full native geometry updates of66.6/421.6/262.2/439.0ms at16/32/64/100 people (five warmups,30 samples). This calls unchanged native functions with independently allocated geometry; it excludes rendering and bypasses the unchanged native transport limits. Nonmonotonic results indicate browser JIT/allocation effects; do not derive a linear scaling model or label these native scene FPS. [Raw final audit](browser/native-cpu-baseline-1790485643971.json).

The new renderer uses two instanced mesh parts per crowd, shared affine animation textures and41 total scene draw calls;64 people draw3,156,332 triangles. It performs no per-frame CPU mesh deformation. Actual WebGL2 transform feedback over three blend cases and255,366 scalar comparisons agreed with native CPU positions to2.16e-7m (tolerance1e-5m). Normals are approximate. No LOD/true shadows; contact blobs only.

## Navigation and motion evidence

| People | Crossings completed / scheduled | Post-separation overlapping pair-steps | Detected deadlocks | Min disc clearance (m) | Actual max acceleration (m/s²) | Low-foot speed proxy p95 (m/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 / 32 | 0 | 0 | 0.0249 | 28.975 | 0.973 |
| 32 | 64 / 64 | 0 | 0 | 0.0258 | 24.597 | 1.089 |
| 64 | 128 / 128 | 0 | 0 | 0.0099 | 38.784 | 0.958 |
| 100 | 199 / 200 | 0 | 0 | 0.0014 | 34.971 | 1.032 |

Seed42;120s,30Hz solver exported at15Hz. Two cycles:15s wait,38s walk,7s clearance. Diagonal/cardinal routes, staggered launch, speed/gait/phase/height/color variation and small groups. The AI plan actually controls route/style/speed/group distributions. Navigation is computed offline and replayed; the renderer runs live. No adaptive interactive obstacles are claimed.

At64, all128 scheduled crossings complete. Residual separation resolves624 preprojection overlap pair-steps, accumulating116.523m correction across the whole crowd; max single-iteration correction4.85cm. This can look abrupt: actual max root acceleration38.784m/s² even though commanded acceleration is bounded. Group p95 maximum spacing2.306m (max4.16m); groups can spread. At100,199/200 scheduled crossings complete within120s. Zero red-phase interior samples and root-boundary violations in all four. These are proxy/seed results, not formal avoidance guarantees.

[Scene geometry checks](scene-proxy-results.json) found zero sampled body-cylinder contacts with generated solid bounds and continuous root-disc support on the foundation. They do not prove limb/body mesh clearance or sole contact.

Foot sliding remains visible and measured. At64, low/vertically stable ankle/toe endpoints have aggregate XZ-speed p95=0.958m/s, max6.792m/s; endpoint bout drift p95=0.523m. Ankle-only p95 left/right is0.151/0.116m/s. Low swinging toes can qualify, so this is a contact-candidate proxy, not a physical contact measurement. All rendered idle/walk blends are included and candidate selection never filters by horizontal speed. [Full gait metrics](gait-64.json), [actual JS/Python parity](gait-renderer-parity.json).

Visual review covered the full120-second sequences in street and overhead views, with additional timeline contact sheets. Waiting groups, staged departures, opposing/diagonal flows, clearance and end holds are visible. Dense encounters still turn sharply; clothing/appearance diversity is limited and repeated gaits are recognizable. The first recording looped at120s; the final recordings hold the end pose. No claim of production-quality crowd realism or frame-by-frame physical correctness.

## Sources, latency and reproduction

One real AI planning call took3.818s. Seven fresh, serial Core horizons took0.816–0.963s each,6.059s total warm inference. Two continuous walking sources plus idle/start/stop source clips are archived with requests, seeds and hashes. Rendering uses reviewed relaxed/brisk walks and idle; generated start/stop clips are **not played**, and transitions use a procedural speed blend limited to1.8 blend units/s. Casual maps to relaxed. Rejected cached loops and unsmoothed metrics remain labeled.

The crossing repositions actual generated city building/lamp parts and adds explicitly authored roads/sidewalks/stripes. This is not a newly generated Tokyo recreation. One authored Xbot mesh is tinted/scaled across people. [Asset provenance](assets/README.md), [source archives](generation/), [recorded AI plan](plan.json).

Offline trajectory solve wall times were1.242/1.932/4.512/8.859s for16/32/64/100; p95 steps0.669/1.139/3.288/4.952ms. These are source-generation measurements, separate from browser render timing. No new Pods rented; used an existing coordinated Core lane and released it.

See [replay/rebuild commands](../../docs/CROWD-CROSSING.md) and [code hashes](code-versions.json). The only post-benchmark runtime configuration change moves Vite dependency cache into the isolated checkout; renderer/assets are unchanged. All four complete trajectory files, raw source NPZs, generated background metadata and source motion arrays are included. Prepared playback requires no service credentials or model worker.

Validation:1,187 Python tests and37 existing client tests pass; TypeScript and isolated Vite build pass. Independent integration review found and resolved gait-selection/metric-blend mismatches. Main Studio files, pair protocol and live demo were not modified. This is a draft milestone for review, not an integration/merge approval.
