# Milestone 2 review — real generated G1 motion

## Complete

Existing stage and supplied G1 retained, with real Pod inference, one text instruction, statuses, local play/pause/reset/camera, optional camera following, history-conditioned subsequent instructions, and stale-result rejection. Original standalone preview remains unchanged and recorded mode remains available. Floor height was brought level with the platform so generated root movement can leave the circular stage without dropping onto a lower floor.

Working private URL: http://YOUR-MINI.YOUR-TAILNET.ts.net:2334/ (Tailscale required). Choose Live ARDY, type an instruction and Generate next 4 seconds. Change the instruction and generate again during playback. Recorded preview selects the original non-AI shadow-boxing clip. Detailed restart/shutdown instructions are in ../README.md.

## Verified model path

Official ARDY source revision: 693f74d13b3d04a0a22ce127ee79c929dd89756b.
Downloaded checkpoint snapshot revision: 059b8007df0ba194a006a877b59a563955ac7b70.
Checkpoint: nvidia/ARDY-G1-RP-25FPS-Horizon52. Actual loaded model asserted G1 g1skel34, 34 joints, 25 fps, horizon 52, four frames/token, 414 features. Uses official CUDA load_model with the local Llama/LLM2Vec text encoder, fresh text encoding, two autoregressive_step calls and motion_rep.inverse. No prerecorded substitute, fixed seed, prompt cache, human skeleton substitution or custom retargeting. Client validates request ID, model, shapes, finiteness and proper rotation matrices.

Existing RTX 6000 Ada Pod, Python 3.12.3, torch 2.14.0+cu130. Access to the required model/encoder was available. No gated terms accepted and no paid resources provisioned. Exact packages: pod-packages.txt. Backend listens only on Pod loopback and requires bearer authentication; verified an unauthenticated health request returns 401. SSH forwards only Mac loopback; viewer is shared privately with Tailscale.

## Measured performance

- Cached service/model loading: 70.28 seconds. This excludes initial environment installation and weight downloads; those are setup work, not interaction latency.
- First fresh request: 1.70 seconds generation, 2.18 seconds client round trip (before warm-up).
- Sustained real inference: 24 requests, 99.995 seconds, zero failures, six distinct instructions repeated over four sequences with freshly sampled output. Each request generates 4.16 seconds. Source measurements: backend-soak.json.
- Warm GPU generation across that test: 0.185–0.214 seconds, median 0.208.
- Warm client request/transfer/decode round trip: 0.383–0.577 seconds, median 0.482.
- Interactive browser render acknowledgements: 0.48–0.99 seconds in observed successful UI requests. Includes transport and a small screenshot returned after pose submission, so it is an upper bound on command-to-visible-response. Source: live-metrics.jsonl. Browser was on the mini, including tests through the private Tailscale URL; the user's remote MacBook latency remains unmeasured.
- Peak torch GPU allocation: 14.95 GiB; reserved: 15.01 GiB. nvidia-smi reported 15,884 MiB for the process after tests. Backend process CPU RSS approximately 2.00 GiB; no progressive increase beyond a few MiB during the soak.

## Tests and visual evidence

Actually inspected rendered output: upright correctly oriented G1 mesh, raised arms, right-handed wave, walking with root travel and camera follow, and squat. Screenshots: stagezero-live-arms.png, stagezero-live-wave.png, stagezero-live-walk.png, stagezero-live-squat.png. These show real generated output paused for inspection.

Browser checks: new instruction while playback was active; pause/resume; pause while a request was pending; reset while pending; replace a pending walking request with squat; camera orbit after reset. Disconnected the actual SSH tunnel, confirmed generation error while retaining the scene, restored the tunnel through run-live.command, and successfully generated again through Tailscale. The connection-error message was subsequently simplified and the viewer restarted. Original recorded mode remains distinct.

Eight controller regression tests pass. These intentionally use controlled simulated transport to force races/failures, and cover late/stale results, changed drafts, pause during generation, reset cancellation, switching to recorded mode, failure/retry, history slicing and invalid skeleton/non-finite data. They are not evidence of AI output; that evidence comes from the real backend and browser tests above.

Continuation boundary mean joint displacement ranged 0.4–29.3 mm (median 2.6 mm) in the soak. This is a numerical continuity check, not proof of perfect animation quality or physical contact. Generated minimum joint height reached -1.2 cm in that test.

## Limitations to review

- Complete 4.16-second segments, not streaming. The actor holds its displayed pose during generation. Manual Generate extends the performance; there is no continuous autonomous generation.
- New instructions condition the next motion but do not force instant action switches. The walking transition inspected began moving after roughly two seconds. Stop/turn and other text instructions are probabilistic, not hard constraints.
- No physics/contact solver. Minor foot sliding or floor penetration can remain. Official G1 postprocessing is not applied; it is disabled in the upstream demo path for this rig.
- Reset/first generation starts a fresh take; the initial pose can differ from the recorded reference. A replacement within the first three generated frames has insufficient token history and also starts fresh.
- One shared actor/state across browser tabs. Latest submitted instruction wins. Cancellation prevents playback of stale results but an already-running GPU operation may finish before cancellation is noticed.
- The sustained inference test covered about 100 seconds, not an overnight soak. Hardware stereo/glasses and remote MacBook timing were not tested.
- Stopping the application/backend does not stop Pod billing. The user controls Pod shutdown.

## Review checkpoint

Milestone 2 is ready for review. Evaluate motion quality, instruction response, camera following, and the segment-based interaction. Recommended next milestone, only after approval: Milestone 3, verify the actual VITURE device and stereo/head-tracking path. No glasses/directing/timeline/multi-actor work has started.
