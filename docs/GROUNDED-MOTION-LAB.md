# Native-motion experiments after the rejected swing

The web-swing presentation was rejected on visual grounds. Its rigid posing,
assisted flight, and procedural character appearance were not convincing. Passing
constraint tests did not make it an acceptable animation result. It remains a
diagnostic experiment; the original studio and concurrent character/background
work have not been replaced.

This isolated lab tests a different approach: play the complete generated body
motion on the existing textured civilian and ranger meshes, in the actual
existing generated Market Square scene. No authored body pose substitutes for
the model output. Character retargeting is explicit: an approximate 17-bone rig,
fixed limb lengths, optional two-bone arms following each native wrist trajectory,
and one disclosed floor calibration per clip. Selected solo clips additionally use explicit vertical boot-clearance IK: only
penetrating near-ground boots lift, with fixed pelvis, original ankle x/z and
unchanged airborne poses. This reduces floor penetration but does not lock feet
or simulate friction. A continuation reuses the same mode and floor
calibration, preserving the rendered prefix as well as the source frames.

## Run locally

From a fresh checkout, initialize the skeleton submodule and install the local viewer dependencies (no model weights or private recorded CSV are needed for this lab):

```sh
git submodule update --init --recursive
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-live.txt
(cd studio_client && npm ci)
```

Then:

```sh
./run-motion-lab.command
```

Open `http://127.0.0.1:2361/`. Bundled samples, meshes, and the generated scene work
without a model connection. New directions require the existing authenticated
realtime service. Set `STAGEZERO_REALTIME_URL` and `STAGEZERO_TOKEN_FILE` to that
service and its local token file. No credential is bundled. `STAGEZERO_PYTHON`,
`STAGEZERO_THREE_DIR` (the `three` package directory), and `STAGEZERO_MOTION_PORT`
can override the local installation paths.

This workspace's command uses:

```sh
STAGEZERO_PYTHON=/Users/jakob/Desktop/Shellhacks/.venv/bin/python \
STAGEZERO_THREE_DIR=/Users/jakob/Desktop/Shellhacks/studio_client/node_modules/three \
STAGEZERO_TOKEN_FILE=/Users/jakob/Desktop/Shellhacks/.runtime/api-token \
./run-motion-lab.command
```

The prototype binds only to localhost. Existing private routes and model workers
are preserved. No additional Pod was rented.

## What is being compared

- ARDY Core: native single-actor dance, guard/dodge, martial combination, stumble,
  and celebration prompts. The celebration did not actually jump; the help-up
  scene below did not actually help someone up. Those are failed semantics.
- InterGen: paired shove/dodge, embrace, help-up and dance. The help-up/embrace
  had substantial body overlap. Dance gives coordinated movement and intermittent
  hand proximity; it is not a secure continuous grasp. Retarget IK improves wrist
  placement but makes some elbow poses more awkward. It is a research comparison.
- Kimodo SOMA RP v1.1: actual inference on the existing GPU, with the Llama text
  encoder on CPU. Raw SOMA77/30fps files are preserved separately. The named
  SOMA-to-Core27 adapter resamples rotations on SO(3) and validates native FK
  before publishing 20fps review files; it is not the original 77-joint output.

InterGen is **CC BY-NC-SA 4.0**, unsuitable for silently incorporating into a
commercial production path. ARDY and Kimodo code are Apache-2.0; their model
checkpoints use NVIDIA Open Model terms. Kimodo's text encoder also has the
upstream Llama model's separate terms. This lab does not redistribute checkpoints.
The generated characters have no finger or facial rig, and the approximate body
rig retains source mesh and texture defects.

## Realtime contract

The local controller requests an actual Core continuation from the last 40 native
history frames. A direction commits a boundary about 60 frames (three seconds)
ahead, preserving every earlier frame and replacing only unplayed future frames.
The native 330-feature history is retained when available. Enable **LIVE**, play a
Core clip, and submit a next action. LIVE replenishes the buffer using the last
submitted action; typing alone does not change it. A new submission during an
active request replaces the one queued instruction. Turn LIVE off to finish the
current buffer. The lab caps a session at 32 requests and preserves saved outputs. The renderer keeps
playing its committed prefix while a new result loads; it swaps complete fitted
arrays atomically. The live fetch transfers only the new tail. The server reuses
the SHA-verified fitted prefix and retargets only newly generated frames, so work
does not grow with the whole played history. If generation is late it holds the final committed frame and
measures that hold rather than replaying old motion or teleporting to a new clip.

Live continuation is enabled only for native solo Core clips. Paired clips are
replay-only because independent Core actors drift through each other. Kimodo
clips are replay-only because their SOMA proportions fail Core history validation.
These failures are checked in the UI and server.

Successive actors in Core are generated independently. An InterGen pair sample
contains coupled interaction, but a subsequent Core continuation does not gain
pair-aware contact constraints. Kimodo is also a single-actor model. Neither
means arbitrary live multi-person physical interaction has been solved.

## Reproduce and inspect

`experiments/grounded_model_probe.py` generates the initial six service samples.
`experiments/grounded_quality.py` reports articulation, inferred foot sliding,
contact proximity and conservative body overlap without emitting an aggregate
visual-quality pass. `experiments/kimodo_probe.py` records bounded official-model
inference with a per-process GPU allocation limit. `grounded_soma.py` converts
saved SOMA files explicitly, preserving their hashes in the converted metadata.

Run the focused checks with:

```sh
python -m unittest test_grounded_character test_grounded_soma \
  test_grounded_quality test_grounded_server
```

Record captures the actual browser canvas, including its controls and live status.
The raw WebM, a telemetry report, actual generated NPZ files, and job reports are
saved under `.runtime/grounded-review/`. Independent samples are not represented
as one continuous live scene. A video montage, if made, must identify its cuts.

## Production work still required

Use a production character rig with fingers, clavicles and facial controls;
retarget contact trajectories with anatomical limits; validate scene collisions
on the rendered body; use a trained physics controller or authored physical
skills for jumps, swings and carrying. Kinematic text-to-motion plus visual
collision checks is not a physical simulation. Realtime pair reaction needs a
pair-conditioned streaming model or an explicitly synchronized skill system.
The supported experiment here is grounded body choreography, not the original
Spider-Man carry/landing/kiss milestone.

Primary references: [ARDY](https://github.com/nv-tlabs/ARDY),
[Kimodo](https://github.com/nv-tlabs/kimodo),
[InterGen](https://github.com/tr3e/InterGen), and
[Human-X](https://github.com/humanx-interaction/Human-X-Interaction).
Human-X's public repository still marks its physics tracker as coming soon;
its paper's complete system is not a currently available drop-in replacement.

## Saved run and recovery

Each completed continuation is an atomic NPZ result; its parent remains unchanged.
Failed generation and retargeting never publish a new playable result. Offline
replay does not require a token or running Pod. To load the saved actual live run:

```sh
./run-motion-lab.command --clips grounded_assets/clips --clips review/grounded-motion/saved-live
```

The saved `final-live.npz` is the final corrected 980-frame run;
`live-solo-final.npz` preserves the earlier 876-frame diagnostic run. Select
the clip with **980 frames** for the corrected run.
This is replay of saved generated frames; starting LIVE again requests new Core
output. The earlier diagnostic bow ends bent over. The final run includes a separate
upright-standing request and reaches that ending.

Read [GROUNDED-MOTION-RESULTS.md](GROUNDED-MOTION-RESULTS.md) for measured results,
actual captures, rejected paths and resource use.
