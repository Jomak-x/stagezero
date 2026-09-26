# Grounded motion: measured replacement experiment

The original swing presentation is rejected. Rigid body posing, assisted flight
and the procedural superhero mesh did not meet the visual goal. Its passing
constraint checks were insufficient. This work preserves that failure as a
research artifact and tests actual generated body motion on the existing textured
civilian, in the existing generated Market Square.

This is a runnable isolated research lab, not a production release or a completed
Spider-Man/carry/kiss scene. The studio and concurrent character, object and
background work remain untouched.

## Final corrected live run

[Watch the continuous final run](../review/grounded-motion/final-live.mp4): all
49 seconds of generated motion and its ending are shown. Only the later paused
idle tail is removed. [The complete uncut 69-second capture](../review/grounded-motion/final-live-full.mp4)
and [capture hashes](../review/grounded-motion/capture-provenance.json) are preserved.
This run uses vertical boot-clearance retargeting and tail-only live transfer.
The visible canvas includes the next-request text and submission state.

| Final-run measure | Observed result |
|---|---:|
| Real Core continuations / saved frames | 12 / 980 at 20 fps |
| Command to first new drawn frame | 2.982–3.116 s |
| Recognizable sustained bow after its command | estimated 6.66 s; 10-frame confirmation 7.11 s |
| Upright posture with hands lowered after its command | estimated 14.37 s; confirmation 14.82 s |
| Generation-buffer holds | 0 s |
| Render callbacks above 100 ms | 0 / 4,141 in the full capture |
| Render callback p95 / maximum | 18.6 / 33.2 ms |
| Displayed raw and fitted prefixes | exact across all 12 continuations |
| Measured mesh floor penetration | none in these 980 frames |
| Minimum conservative clearance from generated props | 1.828 m |
| Inferred sole sliding p95, left / right | 0.189 / 0.230 m/s |
| Final two-second root drift | 6.1 cm |
| Total serialized live transfer reduction | 85.9% |

**The approximately three-second number is not semantic response latency.** Bow
and standing estimates use explicit native-pose thresholds anchored to logged
first-drawn boundaries; they are not direct semantic event timestamps. The model
can retain its previous guard posture after receiving a new action. The final
posture is upright with lowered hands, but it is not perfectly still. Fingers,
facial acting, friction and physical contact remain unresolved.

The final retarget keeps the pelvis path and existing ankle x/z. Only a penetrating
near-ground boot is lifted using fixed-length leg IK; airborne poses are unchanged.
The correction is causal and uses the same frozen floor calibration in later
segments. It does not imply physics simulation or perfect ground contact. An
independent full recomputation exactly matches the optimized cached result.
Tail transfer and prefix reuse replace repeated loading/fitting of the full
history. The last response falls from 10,663,120 to 877,718 serialized bytes.

[Final telemetry](../review/grounded-motion/final-live-telemetry.json),
[independent geometry/continuity/semantic audit](../review/grounded-motion/final-live-audit.json),
and [saved final motion](../review/grounded-motion/saved-live/final-live.npz) are bundled.
The root agent reviewed the complete motion through the actual browser capture
and contact sheets. **312 regression tests pass**; JavaScript syntax and diff
checks pass. These checks certify the stated mechanics, not cinematic quality.

The intermediate corrected run had three render gaps, maximum 433 ms, while the
full regression suite also ran. After tail-only transport and without that
concurrent workload, the final capture above had none. This is not a controlled
benchmark proving which change caused each earlier gap.

## Earlier uncorrected interactive run (retained failure evidence)


[Uncut 48-second browser capture](../review/grounded-motion/live-solo-uncut.mp4)
shows native Core martial motion, a live dodge/block request, and a queued bow
request. The final saved motion contains 876 frames at 20 fps (43.8 seconds);
the recording also includes its final paused view. It is one actual browser run,
not a montage of independently generated clips. The MP4 transcodes the full raw
WebM, adding one bottom pixel for H.264 compatibility, without cuts or speed changes.
The actual canvas includes playback controls, model prompt, frame counter and job
status; the separate HTML text-entry row is outside the captured canvas.

| Measure | Observed result |
|---|---:|
| Actual native continuations | 11, all completed |
| First submitted action to first new rendered frame | 2.982 s |
| Mid-playback dodge/block request to first new frame | 2.998 s |
| Queued bow request to first new frame | 4.049 s, including 1.051 s queue wait |
| Job-ready latency, all requests | 1.328–2.116 s |
| Measured generation-buffer holds | 0 s |
| Renderer callbacks over 100 ms | 0 / 2,861 |
| Renderer callback p95 / maximum | 18.6 / 100 ms |
| Loading callbacks excluded from this run | 0 |

These latencies measure the first frame of the new generated segment, not the
moment a requested gesture becomes recognizable. Renderer callback gaps are
not a guarantee of every display refresh or encoded video frame. There is a
three-second committed playback lookahead by design. Native body poses are
sampled at 20 fps; the canvas recorder targets 30 fps.

The independent mesh audit found **8.22 cm maximum floor penetration** and 130
frames more than 1 cm below the floor in this original retarget. All 876 mesh
frames clear non-floor prop bounds by at least 1.828 m. All 11 raw and fitted
prefixes are identical, but the largest rendered vertex step at a seam is 20.5 cm;
fast kicks explain much of that motion, not a proof of velocity continuity. This
recording verifies live generation, but fails strict grounded visual quality.
See [the independent audit](../review/grounded-motion/live-solo-audit.json).

The actor actually punches, raises a kicking leg and changes posture during
playback. The ending bends forward but does not finish the requested return to
standing. The dodge does not establish reliable geometric leftward navigation.
Do not describe complete prompt fulfillment as solved.

[Telemetry](../review/grounded-motion/live-solo-telemetry.json),
[contact sheet](../review/grounded-motion/live-solo-contact-sheet.jpg), and
[saved motion](../review/grounded-motion/saved-live/live-solo-final.npz) accompany
the run. Follow [the launch/reload instructions](GROUNDED-MOTION-LAB.md).

A provenance bug found after capture inherited parent request fields, including `seed` and `frames`, in the
saved NPZ metadata. The controller actually requested seeds 1702 through 1712 in
this fresh session; the independent report identifies this derivation. The raw
capture archives remain unchanged. New outputs now record the actual request
seed and retain the parent seed separately; regression coverage checks this. New request IDs, frame counts and chunk metadata also describe the actual continuation.

## Actual model comparison and discarded approaches

Kimodo RP v1.1 was run on the existing GPU, with its text encoder on CPU. Its
martial take gives a legible high kick and a return to guard. This is a selected
sample, not evidence that it always outperforms Core.

[Separate Kimodo martial replay capture](../review/grounded-motion/kimodo-martial-replay.mp4)
is explicitly a replay of real generated output, **not live Kimodo steering**.
See the [same-prompt comparison and visual decisions](../review/grounded-motion/model-selection.md).

- **Live paired continuation rejected:** independent Core actors pass through
  each other. The test reached 5.2 cm root separation and about 29.8 cm overlap
  in the conservative torso-capsule proxy. Smooth playback does not fix contact.
- **InterGen help-up and embrace rejected:** substantial body overlap and missing
  requested action. Paired dance has intermittent hand proximity, not a secure
  hold; wrist IK also produces awkward elbows. Pair samples are replay-only.
- **Direct Kimodo-to-Core continuation rejected:** SOMA proportions are not valid
  Core history, despite sharing a reduced 27-joint layout. Backend skeleton
  validation rejected the real request. The UI and server now prevent this path.
- **Energetic hip-hop not selected:** both tested models give modest gestures and
  footwork rather than the requested energetic dance.
- **Core celebration jump rejected:** no convincing airborne jump.
- **Kimodo cartwheel rejected after revised retarget trials:** native-world wrist
  targets reduced the original 22.6 cm hand-floor penetration to 7.1 mm, and boot
  clearance removed the landing penetration. The inverted pose still lacks
  convincing palm support and a local knee adjustment adds an 8.2 cm frame step.
  It is preserved under `grounded_assets/rejected`, outside the normal showcase.
- **Kimodo floor sweep and shoulder roll rejected:** neither performs the defining
  requested action. The spinning-kick prompt yields a useful high kick, but not
  a clearly completed spin and two-punch combination.

The real generated market snapshot contains 52,392 triangles. Its generated props
came from the existing background pipeline; their market layout is that pipeline's
deterministic composition. It is not a newly invented stand-in or a photoreal set.
The source SHA, metadata and every bundled mesh/motion hash are in
[the asset manifest](../grounded_assets/artifact-manifest.json).

## Failure and reload verification

A real unreachable-service request failed without changing its source or publishing
a playable result. The same controller then recovered against the existing service:
240 total frames were ready in 3.405 seconds, with the 160-frame source prefix exact.
A fresh controller loaded the saved 876-frame run with identical array hashes and
its original floor calibration. [Recovery evidence](../review/grounded-motion/recovery-results.json).

## Resources and licensing

No extra Pod was rented. The existing RTX 6000 Ada workers were preserved. Kimodo
had about 15.5 GB free VRAM available, a 10% process allocation cap, and peaked at
about 1.24 GB allocated GPU memory with CPU text encoding. Eight-second generation
was approximately 3.3–3.5 seconds after a 58-second cold load. The source weights
occupied about 17.55 GB of RAM-backed storage; GPU allocation is not total system
memory use. This was a bounded offline probe, not a resident production service.
Exact commands, timings, source commit and limitations are in
[the resource note](../review/grounded-motion/kimodo-resources.md).

ARDY and Kimodo code use Apache-2.0; their weights have NVIDIA Open Model terms.
The text encoder has separate Llama terms. InterGen is CC BY-NC-SA 4.0 and remains
a noncommercial research comparison. No model checkpoint or credential is bundled.

## What production integration requires

The Core buffering and atomic clip replacement path can be adapted to the studio,
with the existing studio's project/timeline APIs and cancellation semantics. Keep
the native motion history and commit boundary contract; replacing them with a
fresh unrelated clip produces a visible seam.

Kimodo needs a separate worker and native SOMA motion contract, or a verified
canonical conversion plus constrained generation. Its current sample generation
is neither causal streaming nor a tested live history continuation. A resident
worker would remove cold-load cost but still needs scheduling and latency tests.

Character quality needs a proper full-body rig with hands, fingers, clavicles and
facial controls, plus better skin weights. The current approximate 17-bone rig
cannot show closed fists or a convincing kiss. Carrying, hand-supported acrobatics,
and scene interaction need contact-aware retargeting and either a trained physics
controller or authored physical skills. Arbitrary paired interaction needs paired
training data/control; two independent solo models are insufficient.

Do not merge the lab into the studio as a finished movement system. Keep the
stronger generated actions, preserve the failed experiments, and review the actual
clips before choosing the next production milestone.
