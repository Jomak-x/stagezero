# Two-character choreography review

> User review: REJECTED. Excessive separation and mistimed movement make this unsuitable as convincing interaction. Preserved as failed evidence; superseding experiments are in progress.

The current integrated **Pose duet** preset uses the bounded `pose_duet_v1`
profile: two independently generated Core actors perform an eight-second staged
rehearsal with an opening, a kick/duck exchange, reversed roles, and a shared
victory pose. It adds shared timing and reviewed pose targets to the existing
Studio. The first duck in the selected seed is late; the result is not a
contact-aware fight or a finished motion-quality solution.

Watch the [selected UI-generated video](../review/two-character/ui-pose-duet-capture/performance.mp4)
(seed 6201), or inspect its [contact sheet](../review/two-character/ui-pose-duet-capture/contact-sheet.png).
The [CLI trial capture manifest](../review/two-character/pose-duet-spaced-capture/capture-manifest.json)
records all 160 native frames at 20 fps, at 1280 × 720, in the saved 50-object
city scene. Each frame was rendered from the saved archive in order. There is
no pose splicing, interpolation, duplicated-frame slow motion, or postprocessed
foot locking. This is an offline replay of actual generated motion, not a
recording of live generation latency.

## Use Together in the Studio

Connect the Studio to the existing authenticated Core worker as described in
[Scene direction](SCENE-DIRECTION.md#launch-against-an-existing-service), then:

1. Open **Motion → Scene direction · Core → Together · experimental**. Select
   **Pose duet** in **Shared preset**.
2. Press **New two-person performance**. This activates Core, backs up an
   existing populated Core project, and searches the current scene for two
   supported placements at least 2.25 m apart, facing each other. A blocked
   scene fails explicitly; use a spacious layout.
3. Optionally choose **Swap actor roles**, then **Preview shared preset**.
   Review the shared beats and both roles before generation.
4. Press **Generate shared sequence**. The sequence appends four complete
   two-second windows to the native timeline. Inspect it with **Play**,
   **Pause**, **Restart**, and the frame slider. Cancel or retry through the
   existing Core controls if generation fails.
5. Use **Save exact Core project + download**. Reopen through **Core projects →
   Open saved Core project** or **Open Core project file**. Loading the archive
   does not request new motion.

A fresh UI run completed all 160 frames; its
[archive](../review/two-character/ui-pose-duet.core.stagezero.npz) also loads as
`pose_duet_v1`. Shared plans become stale when relevant scene, cast, or direction
state changes; preview the current plan again before submitting it.

For exploratory AI direction, enter **Shared direction for AI**, press **Plan
shared scene with AI**, inspect the returned plan, then press **Generate shared
sequence**. Planning runs asynchronously through the existing Neon gateway.
It proposes 2–8 shared beats, at most 16 seconds, with complementary per-actor
prompts. Invalid output fails without silently switching to an offline preset.
The AI planner does not generate, execute, or verify the resulting motion.

## What the reviewed profile adds

`pose_duet_v1` has four fixed two-second beats and a pinned native cue asset,
`assets/core-motion/duet-cues.npz`, checked by SHA-256. Its kick, duck, and victory
cues supply four target frames at the end of each applicable generated window.
The model still generates every committed frame. Targets are aligned to actual
actor roots and partner direction; the shared victory uses a common audience
heading. The profile conditions subsequent windows on four native history
frames and advances the seed per window. Full-body target requests omit
separate root targets because combining them failed on the current worker.

Only the reviewed schedule or a complete role swap uses this profile. Arbitrary
AI text cannot select custom pose assets, supply scene mutations, or acquire
reviewed status by naming the recipe. AI candidate plans remain free
choreography; pose cues are a separate, fixed implementation.

The profile requires **2.25 m initial root separation**. Before committing each
profile window, a **0.15 m minimum cross-actor joint-distance proxy** supplements
existing scene-body, root-disc, and authored-floor checks. A rejected window
retains earlier committed motion. Joint centres and conservative scene proxies
are not mesh collision tests, full foot locking, or physical contact simulation.
Both actors remain independently conditioned, without shared balance or impact
response.

## Measured results and limitations

The final Python verification passed **731 tests**; see the
[test log](../review/two-character/python-tests-final.txt). Tests validate code
contracts, not animation appeal. The existing warm Core worker completed the
following serial trials without renting another Pod:

| Measurement | Seed 6201, selected | Seed 6202 |
| --- | --- | --- |
| Complete generated motion | 160 frames / 8 s | 160 frames / 8 s |
| GPU jobs | 4 | 4 |
| First new committed window | 0.852 s | 0.927 s |
| Generation loop completed | 3.681 s | 3.790 s |
| Observed post-start buffering holds | 0 s | 0 s |
| Exact saved/reloaded native arrays | Passed | Passed |
| Minimum cross-actor joint distance | 0.3772 m | 0.7362 m |
| Maximum root step at a window boundary | 0.0432 m | 0.0396 m |
| Maximum joint step at a window boundary | 0.3428 m | 0.2133 m |
| Maximum joint rotation change at a boundary | 43.80° | 24.26° |

Sources: [6201 report](../review/two-character/pose-duet-spaced-6201.json) and
[6202 report](../review/two-character/pose-duet-spaced-6202.json). Both reports
show zero sampled scene-overlap frames, zero root-disc overlap frames, and
supported root footprints. Their top-level `conditioning_history_frames: 40`
is the generic trial-harness setting; the recipe metadata and production
adapter use four history frames for this profile.

First-commit time measures the controller receiving motion, not user-command
to visible-action latency. Generation completion is not playback completion.
The capture's 79.14 ms median WebGL render round trip measures screenshot
capture, not browser playback frame rate. Neither these timings nor two seeds
establish broad prompt reliability.

The selected clip's first boundary, frame 40, has a **0.343 m maximum joint
step and 43.8° maximum joint rotation change**. Small root steps alone would
conceal this issue. Its first duck is late, and feet still slide: height-based
low-foot speed proxies have medians of 0.100/0.050 m/s and 95th percentiles of
0.956/0.733 m/s for the two actors. These samples do not establish actual foot
contact. The accepted spacing improves clearance; it does not solve timing,
transition continuity, or planted feet. Raw trial reports retain their original
`visual_review: pending` field; the selected replay and this narrative record
the subsequent review rather than changing historical measurements.

## Preserved rejected attempts

The [earlier integrated 1.8 m trial](../review/two-character/pose-duet-integrated-6201.json)
completed but was rejected for handoff: minimum cross-actor joint clearance was
only **0.0621 m**, with three frames below 0.12 m, despite passing root-disc
separation. This prompted the larger initial spacing and joint proxy gate.
Its archive remains evidence of the earlier behavior, not an alternate accepted
demo. Current code rejects that recipe at 1.8 m before generation.

The [combined root/full-body target experiment](../review/two-character/staged-reference-finish.json)
failed with a CPU/CUDA tensor-device mismatch in the worker. Its first 200
frames and failure report were preserved. A
[subsequent experiment](../review/two-character/staged-reference-finish-v2.json)
completed after removing the redundant root constraints. The integrated profile
uses that request separation; no worker restart or extra Pod was needed.

Earlier text-only, AI-plan, history-length, and reference-finish candidates
remain under `review/two-character/`, including
[the first AI candidate](../review/two-character/feint-ai-v1.json) and
[the revised AI candidate](../review/two-character/feint-ai-v2-h4.json).
Those trials informed the fixed pose-cue approach; producing a plausible plan
was not evidence that the desired motion occurred.

## Reproduce generation and capture

Run from the repository root with the installed `.venv`, built Studio client,
existing Core service/tunnel, and authorized existing Core token file. Set
`CORE_REVIEW_TOKEN_FILE` to that file's absolute path; the commands read it
without printing its contents. Keep the service's existing owner informed and
run trials serially. These commands write new results under `.runtime/` and
preserve the reviewed evidence.

```sh
export CORE_REVIEW_TOKEN_FILE=/absolute/path/to/existing-core-token

.venv/bin/python experiments/trial_core_choreography.py \
  --preset pose_duet --seed 6201 --name pose-duet-rerun-6201 \
  --separation 2.25 --facing each-other \
  --url http://127.0.0.1:8769 --token-file "$CORE_REVIEW_TOKEN_FILE" \
  --scene review/scene-integration/live-city.json \
  --output .runtime/two-character-rerun

.venv/bin/python experiments/trial_core_choreography.py \
  --preset pose_duet --seed 6202 --name pose-duet-rerun-6202 \
  --separation 2.25 --facing each-other \
  --url http://127.0.0.1:8769 --token-file "$CORE_REVIEW_TOKEN_FILE" \
  --scene review/scene-integration/live-city.json \
  --output .runtime/two-character-rerun
```

Each trial saves its exact plan, report, and committed Core archive, including
partial output on failure. A completed rerun still needs visual review. To
recapture the selected saved seed without inference, use an unused local port
and a new output directory; capture refuses to overwrite existing artifacts:

```sh
.venv/bin/python experiments/capture_core_performance.py \
  --archive review/two-character/pose-duet-spaced-6201.core.stagezero.npz \
  --output-dir .runtime/pose-duet-recapture --port 24892 \
  --camera-position 1.4 1.9 4.3 --look-at 0 1 0 --fov 48 \
  --width 1280 --height 720 --sheet-frames 24
```

Open the printed loopback URL in a browser, enter Studio, and click **Start
exact Core capture** before the connection timeout. Keep the browser connected
until completion. `ffmpeg` must be on `PATH`. The output includes
`performance.mp4`, a contact sheet, first/last frames, and a hash manifest.

To create and trial an AI candidate separately, configure the existing private
`.runtime/objects.env` gateway settings first:

```sh
.venv/bin/python core_choreography_ai.py \
  'An eight-second solo shadowboxing call and response, ending in a shared celebration.' \
  --seed 6201 --output .runtime/ai-duet-candidate.json

.venv/bin/python experiments/trial_core_choreography.py \
  --plan .runtime/ai-duet-candidate.json --name ai-duet-candidate \
  --separation 2.25 --facing each-other \
  --url http://127.0.0.1:8769 --token-file "$CORE_REVIEW_TOKEN_FILE" \
  --scene review/scene-integration/live-city.json \
  --output .runtime/two-character-rerun
```

The plan's saved seed controls `--plan` trials. The planner uses the configured
asset-planning model, defaulting to `gpt-6-astra` for Neon, with an explicit
`--model` override available. Gateway planning usage and Core GPU generation
are separate operations.

## Next work

Measure cue timing explicitly, beginning with the late first duck, and reject
large joint/rotation discontinuities rather than relying on root continuity.
Compare more seeds and role swaps before expanding the reviewed preset set.
Add contact-aware foot stabilization only with whole-body and floor checks so
planted feet do not hide knee or root artifacts. For actual partner contact,
a controller with shared actor state, contact timing, and physical feasibility
is still needed. Broader AI direction should select among reviewed motion
recipes only after their outcomes and transitions have been measured.
