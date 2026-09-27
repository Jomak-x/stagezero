# Verification log

2026-09-27, isolated `codex/demo-environments` checkout, merged main `184eca0`.

- TypeScript: `studio_client/node_modules/.bin/tsc --noEmit -p studio_client/tsconfig.json` passed after flashmob camera, ground-shadow and music-rig integration.
- Production client: `cd studio_client && ./node_modules/.bin/vite build --config demos.vite.config.mts` passed (16 modules). The ordinary Vite large-chunk warning remains; the JS bundle is831 kB,258kB gzip. Output is isolated under `.runtime/demos-client-build`; the original UI build is untouched.
- `python -m unittest -q test_crowd_demo_cues test_crowd_demo_motion_export test_crowd_demo_gait_metrics test_crowd_demo_terrain test_crowd_gait_metrics`:29 tests passed in 9.597 s. These check source reconstruction, shader-equivalent foot proxies, bounded terrain tests and malformed/rejected paths; they do not assert photorealism or perfect physical contacts.
- Browser starts only after the GPU affine transform-feedback parity check passes against native CPU skinning, tolerance1e-5m. Each profile retains its actual parity result and payload SHA-256 values.
- Nine navigation tests passed in5.655s against the final112-person scenes. The independent navigation suite and all18 final archive checks are recorded in `NAVIGATION.md` and `navigation-metrics.json` after the crossing density correction.

The screenshots and films are from the real local browser at1200×842, pixel ratio1, Chrome154, ANGLE Metal AppleM6. Development recording runs overlapped offline planning on the same computer. The separate count benchmarks are identified independently in `RESULTS.md`; no claim of isolated-hardware timing is made for the development trial videos.

Final combined run:38 tests passed in16.175s after both city masters froze. The four H.264 movies decode fully at1200×842,30FPS. Full takes are119.97–120.00s; the three-scene highlight is89.80s. SHA-256 values, dimensions and decode results are retained in `videos/verification.json`. The city contact sheet was extracted from the encoded film and visually reviewed.
