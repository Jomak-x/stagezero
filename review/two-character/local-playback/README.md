# Local character playback and faster city entry

The paired scene is now loaded once into the browser. Static actor geometry and exact affine motion palettes replace per-frame vertex streaming. Both actors share one local clock; interpolation reuses vertex and normal buffers. Generation still runs on the existing GPU service. This change covers the native paired workflow, including composed Core approaches, not every older G1/Core renderer.

## Verified observations

- Prior pair payload: 680,976 vertex bytes/frame, or 163.43 Mbit/s at 30 Hz before overhead.
- Revised 412-frame sparring payload: 4,956,528 bytes once (plus static scene/mesh data). Natural playback sends no actor mesh or transport frames; explicit controls and capture still send commands.
- Local Mac mini in-app browser: 60 animation callbacks/second, zero reported dropped callbacks in sampled windows; animation CPU work around 3.2–3.4 ms/frame. This is not a GPU benchmark or a measurement of the user's remote MacBook.
- Pausing only the scratch viewer process for 12 seconds left the server timeline at frame 252 while client animation continued through frame 411. Network-free local clock behavior also has deterministic tests.
- Actual Xbot relaxed and fist affine reconstruction: maximum cross-language position error 2.384e-7 m. Skinning fidelity is preserved; this is not evidence of anatomically perfect contact.
- Two full browser captures: sparring 412 frames / 13.733 s; handshake 373 frames / 12.433 s. Each contains every source frame, without cuts or speed edits. Half-speed controls in watch.html are review-only.

## Motion changes

Preview and generation now use continuous entry at 0.85 m/s. Routes have checked rounded corners and a final leg aligned with the interaction. The actors turn during approach; a shorter checked bridge enters the complete native interaction.

| Scene | Previous native start | Revised native start | Native frames retained |
| --- | --- | --- | --- |
| Crossed sparring | 10.70 s | 6.733 s | 210 / 210 |
| Handshake | 8.70 s | 5.433 s | 210 / 210 |

Both actual-model full scenes passed scene joint/segment proxy collision and continuous floor checks. Handshake first hand-distance below 10 cm moved from 10.167 s to 6.900 s because the approach is shorter; the native contact itself is unchanged. Bridge foot travel fell from 27.2 cm to 17.5 cm; feet are still not locked. One toe is 0.44 mm below zero in the handshake bridge.

Visual review accepts these as incremental improvements: less waiting and fewer large settling turns, readable sparring and handshake. Foot glide and an authored pose adjustment remain. Sparring is not verified physical impact. No new Pod was needed, and no existing model worker was replaced.

## Reproduce

Use the existing token file and provider configuration; never commit credentials. The exact handshake command and comparison are in `handshake-continuous/comparison.json`.

```sh
.venv/bin/python experiments/trial_paired_direction.py \
  --request review/two-character/city-meetup/requests/crossed-sparring.json \
  --scene review/scene-integration/live-city.json \
  --token /absolute/path/to/existing/api-token \
  --entry-policy continuous --speed-mps .85 \
  --output /absolute/path/to/new-output
cd studio_client
npm run typecheck
npm run test
npm exec vite build -- --outDir ../.runtime/local-playback-client
cd ..
STAGEZERO_CLIENT_BUILD="$PWD/.runtime/local-playback-client" .venv/bin/python director_viewer.py \
  --reference-only --port 2380 --token-path /absolute/path/to/existing/api-token \
  --native-pair-config .runtime/native-pair-provider.json \
  --objects review/scene-integration/live-city.json \
  --native-project review/two-character/local-playback/sparring-capture/scene.native-pair.stagezero.npz
```

Use Export video in the studio for an exact full-frame WebGL capture. `watch.html` plays both saved captures. Prior failed arch-floor and rejected embrace evidence remain unchanged. InterGen remains research-only under its noncommercial license.

## Checks

976 Python tests passed in the full run; the final readiness/reconnect/capture fixes additionally passed all 27 affected tests. 35 client tests, TypeScript checking, production build, and diff checks passed. The existing large client-bundle warning remains.
