# Realtime integration results — 2026-09-26

The isolated realtime pipeline runs native ARDY Core navigation and object
reach, switches through a native conditioned transition into a single cached
InterGen pair sequence, then resumes Core from canonical pose history. The
existing studio and other character/object work were preserved.

Private live viewer: http://jakobs-mac-mini.tail5a8376.ts.net:2350/

## Measured acceptance

The independent report is [review/realtime-validation.json](../review/realtime-validation.json).
It uses actual GPU jobs, then independently reloads each complete project,
compares exact raw arrays, checks every 40-frame boundary, and verifies that one
shared rigid transform maps the paired source to both committed actors.

| Scene | Seed | Duration | Startup | Post-start underrun | Outcome |
|---|---:|---:|---:|---:|---|
| Gate, handshake, departure | 42 | 30 s | 2.858 s | 0 s | Passage, hand proximity, release, separation, seams pass |
| Staged fight, disengagement | 42 | 30 s | 3.570 s | 0 s | Dodge, guard, push reaction, release, seams pass |
| Console inspection, departure | 42 | 30 s | 0.727 s | 0 s | Approach clearance, native hand target, seams pass |

Warm Core soak: **25/25 requests succeeded**. First two-second window latency
was 0.827 s median, 0.908 s p95, and 1.509 s maximum. Fifth-percentile generated
motion duration / wall time was 2.20. Progressive publication, queued cancellation,
recovery, immutable committed prefixes and exact save/load passed. Model startup
was approximately 76–79 seconds; these warm latency figures exclude model loading.

Handshake: minimum wrist gap 2.3 cm; longest continuous interval within 15 cm
was 1.15 s. Console: minimum wrist-to-target distance 3.0 cm; 1.75 s continuously
within 12 cm. These are measured proximity checks, not a physical grasp solver.

The selected fight has 91.3 cm lateral dodge displacement, a 6.2 cm minimum
hand/forearm guard gap sustained within the guard tolerance for 0.75 s, and
pair separation increasing from 0.766 to 1.512 m in the push-reaction window.
**7 of 24 screened paired samples passed all six action proxies.** An initial
seed-42 fight with a different prompt failed block/push checks; a complete
seed-38 fight passed but looked like loose sparring. The selected prompt-2,
seed-42 pair visibly improves arm engagement and separation, and its full
live scene passes the same gates. Both earlier full runs are retained as
diagnostics. The block and push are staged visual actions, not validated
physical contact; this curation is not evidence that arbitrary fight prompts
are reliably correct.

## What is ready and what is bounded

The authenticated service, bounded buffering, cancellation, exact project
persistence, explicit failure states, and known scene paths are implemented and
tested. Model-generated poses are not replaced with authored motion to make
metrics pass. A render uses the saved native pose arrays at their original rate.

InterGen remains **research-only**, disabled by default for production callers.
Its CC BY-NC-SA license does not make this a commercial paired-motion release.
See [release gates and primary license sources](REALTIME-PRODUCTION-GATES.md).
ARDY Core is the production-profile motion source.

Objects need declared geometry and verified affordances. A solid pillar is not
accepted as a gate. Reach is reference-conditioned contact, without attachment,
grasping, rigid-body physics, or general visual object recognition. Conservative
root-disc overlap during handshake is recorded as a review flag; mesh-level
physical collision is not certified. Full studio adoption remains a separate
integration step because other agents are actively changing that UI.

## Review recordings

See the [three full videos, contact sheets, and browser screenshots](../review/realtime-showcase/README.md).

## Reproduction

See [REALTIME-DIRECTING.md](REALTIME-DIRECTING.md) for the local launcher, isolated
worker, private tunnel, scene presets, and exact project import/export. Default
scene seeds are the verified values above; explicit seed overrides retain the
same outcome gates and may fail honestly.

## Browser verification

The live viewer generated an eight-second gate route through the real Pod:
actual aperture crossing passed and measured prop-overlap frames were zero.
A six-second new instruction appended to that route with its first 160 frames
bit-for-bit unchanged. The private SSH tunnel was then deliberately disconnected;
the viewer displayed a recoverable error. After reconnecting and pressing Retry,
the timeline reached 400 frames with its original 280-frame positions and
rotations unchanged. Browser project upload/playback and downloaded NPZ equality
were verified, including scene geometry and the saved playhead. See
[realtime-browser-validation.json](../review/realtime-browser-validation.json).

Final full local regression suite: **250 tests passed**. The navigation test
includes the actual request validator at headings ±π, following a browser-found
rounding bug that was fixed before the successful route run.
