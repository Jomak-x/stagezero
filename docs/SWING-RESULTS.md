# Live swing prototype results — 2026-09-26

The tested scene runs in one continuous interactive session in the isolated
`codex/web-swing-prototype` worktree. The existing studio, character, object and
background implementations were inspected and preserved. The live app is at
http://127.0.0.1:2360/ on the host Mac. A new private Tailscale route has not been
enabled; that access change is pending user approval.

[Run instructions and production integration work](SWING-PROTOTYPE.md).
[Actual uncut browser video](../review/swing-prototype/interactive-run.mp4).
[Independent measurements](../review/swing-prototype/verification.json).

## Actual recorded test

Run `20260926-062105-a81b85` contains 8,785 logged 60 Hz frames (146.4 seconds).
Its MediaRecorder WebM spans 146.266 seconds, contains 4,389 video packets, and
has a largest inter-packet gap of 50 ms. The MP4 is a full transcode of that
single recording: no cuts, concatenation, speed changes, or offline scene renders.
It includes the actual buttons, command indicators, close carry camera and
profile ending. The scene is stylized and mechanically constrained, not polished
cinematic acting.

I pressed Start swing, Left, Right, Carry MJ, Right while carrying, Land, and
Kiss after landing. The independent checker verifies the actual actor phase at
each command, including airborne pre-carry turns, a fully attached carried turn,
and a grounded ending. It does not accept a mere list of command names.

| Measurement | Final run |
|---|---:|
| Command receipt → first browser canvas draw acknowledgment | mean 57 ms; maximum 94 ms |
| Command receipt → physics application | mean 10 ms; maximum 17 ms |
| Live Core refresh request → response | mean 0.848 s; maximum 1.086 s |
| Live native Core outputs applied and saved | 7/7 |
| Reported engine / browser stalls | 0 / 0 |
| Largest hand-grip error during fully active carry | 7.62 mm |
| Largest joint / root movement per 60 Hz tick | 0.143 / 0.123 m |
| Maximum bone-length drift | approximately 3 × 10⁻¹⁵ m |
| Web intersections outside measured source-anchor terminal allowances | 0 |
| Independent acceptance gates | 17/17 pass |

The browser draws streamed poses at approximately 20 Hz while the controller
steps at 60 Hz. Canvas acknowledgment is a software measurement, not display
photon latency. Joint movement includes deliberate flight and arm motion; it is
not a measured teleport. The video packet timing confirms continuous capture,
not a guarantee that every browser frame is visually distinct.

## Geometry and contact interpretation

The app imports the exact source triangles from the existing generated city
composition: 16 props, seven model-generated buildings, 10,548 triangles.
Roofs come from generated box-face metadata and web anchors are actual upper
facade mesh vertices. [Source hashes and derivation](SWING-BACKGROUND.md).

No torso overlap was measured. Conservative whole-body capsule/AABB sampling
reported up to 2.52 cm overlap near the feet. The worst flagged samples were
then checked against actual source triangles and official CoreSkin vertices:
both were proxy false positives. See the [exact mesh checks](../review/swing-prototype/mesh-check.json).
The checks are sampled geometry tests, not a rigid-body physical simulation or
proof for every possible user command. Decorative roof geometry requires
measured terminal allowances at the attachment building; other buildings get
no such allowance. Both final foot positions remain over the selected roof.

The ending is authored head/hand IK plus a procedural mask reveal, blink and lip
cue. Exact CoreSkin tests keep head surfaces separated by at least 2.22 cm;
the final mouth target centers are 2.42 cm apart before the lip cue. This is a
simple staged ending, not learned facial animation or a production face rig.

## Failed experiments and revisions

- One native Core prompt produced an overhead arm, but no usable aerial arc.
  A different hanging prompt missed the gesture entirely. The exact outputs and
  measurements are retained in `assets/swing-motion`.
- An early live pickup after a turn snapped MJ's heading, moving a hand 0.539 m
  in one tick. Persistent, rate-limited heading and a regression fixed it.
- Reattaching to a behind-the-character anchor stalled travel. Forward anchor
  selection, release and city-boundary turns fixed that behavior.
- Root-to-anchor clearance did not establish hand-to-anchor clearance. Every
  displayed web is now checked from the actual posed wrist, with release/retether
  when occluded; the renderer displays that same straight segment.
- The first ending camera hid contact behind the hero. The final uncut take uses
  a character-relative profile camera. Face placement was calibrated against
  the actual skin mesh rather than nominal joint points.

The final full regression suite passed **276 tests**. Later archive-only changes
seal a capture's report and stop unbounded logging after capture; their dedicated
retention test passes. These do not alter controller or pose output, but the
current server source is not byte-identical to the capture-time server.

## Resource and license boundary

No additional Pod was rented or provisioned. The existing RTX 6000 Ada had
15,473 MiB free at the initial check; all native requests used the existing warm
worker. G1, realtime and character workers were preserved. The account-specific
Pod rate could not be read because the local RunPod CLI lacks an account API key.

ARDY Core checkpoint terms are separate from the Apache-2.0 rig/code. InterGen
and other researched models are not dependencies of this live sequence. See
[model comparison and license sources](SWING-MODEL-RESEARCH.md). The dominant
flight, carry and ending controls are authored; convincing acrobatics, balance,
fingers and facial acting require better authored assets, motion capture or new
contact-aware training data before production integration.
