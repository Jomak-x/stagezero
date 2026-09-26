# Motion interactions: ARDY G1 feasibility and experiment plan

Research note and isolated GPU probe, 2026-09-26. This concerns the pinned
`vendor/ardy` commit `693f74d13b3d04a0a22ce127ee79c929dd89756b`,
StageZero's existing `ARDY-G1-RP-25FPS-Horizon52` checkpoint, and the existing
RTX 6000 Ada Pod. The completed probe demonstrates two **independent** actor
samples in one model batch and a one-actor planar root target. It does not
demonstrate learned two-person interaction, hand contact, or object-conditioned
motion. The [probe script](../experiments/interaction_probe.py) and
[measured report](../review/motion-interaction-probe.json) preserve the exact
cases and failures.

## What the pinned stack supports

ARDY's released model accepts text, normalized explicit motion history, and
optional per-frame `motion_mask`/`observed_motion` tensors. The official demo
generates a 52-frame G1 horizon and may expose constraints beyond that horizon
inside its bounded attention window. The supported constraint families are
planar root positions and heading, full-body keyframes, and end-effector joint
positions/rotations. These are **soft generative conditions**, not hard
collision, reachability, contact, or exact-pose guarantees. [NVIDIA ARDY
overview](https://research.nvidia.com/labs/sil/projects/ardy/), [paper](https://arxiv.org/abs/2607.08741),
[pinned model API](../vendor/ardy/ardy/model/ardy_model.py),
[pinned demo generation](../vendor/ardy/scripts/interactive_demo/generation.py).

The practical constructor path is `Root2DConstraintSet` or a hand constraint
set, then `model.motion_rep.create_conditions_from_constraints_batched(...,
to_normalize=True)`, then `model.autoregressive_step(...)` with masks aligned to
the visible history, 52 generated frames, and any future context. The CLI shows
the same conversion path and the 10-second trained-window cap. At 25 fps with
four-frame tokens, a safe *initial test budget* is at most 248 visible frames,
and `history + 52 + future <= 248`; actual checkpoint/demo settings should be
read at runtime. Place targets at explicit frame indices; beyond-window targets
must be introduced on a later replan. See [constraint constructors](../vendor/ardy/ardy/constraints.py),
[conditioning conversion](../vendor/ardy/ardy/motion_rep/reps/base.py),
[demo window sizing](../vendor/ardy/scripts/interactive_demo/window_budget.py),
and [CLI history budget](../vendor/ardy/scripts/generate.py).

There is an important coupling in the high-level hand classes:
`LeftHandConstraintSet`/`RightHandConstraintSet` expand to G1 hand-chain
positions plus wrist rotation **and** pelvis position, root height, and root
heading at the same keyframe. Their input is a coherent full FK pose array,
not just a three-vector hand target. The lower-level `create_conditions`
supports sparse feature dictionaries, but global joint position conditions
still require a root position condition at the same time. A probe should first
build a valid reference pose/constraint, inspect every nonzero mask channel,
then perturb one reachable hand target. Do not feed an arbitrary desired hand
position with an incompatible root/heading pose. [Pinned constraints](../vendor/ardy/ardy/constraints.py),
[representation fill rules](../vendor/ardy/ardy/motion_rep/reps/ardy_motionrep.py),
[G1 skeleton](../vendor/ardy/ardy/skeleton/definitions.py).

`model.__call__` and the motion representation support batch dimensions and
per-sample constraint lists. `autoregressive_step` also obtains batch size from
text features and accepts batched histories/conditions. A batch of two is
therefore an API-supported *independent sample batch* when both actors use the
same checkpoint, aligned history length, generation horizon, and tensor
shapes. It is **not a two-person model**: actor A cannot attend to actor B's
live latent state. Their relationship must be encoded externally as world-space
targets and timing. StageZero currently serializes requests behind one model
lock, calls two unconditioned 52-frame steps per 104-frame segment, and returns
one actor's arrays; at the time of this probe its endpoint had no constraint or
two-actor contract.
[Pinned batch path](../vendor/ardy/ardy/model/ardy_model.py),
[StageZero backend](../pod_backend.py), [StageZero client](../live_motion.py).

The bundled correction library exposes root/full-body/hand/foot masks and can
alter generated root positions and local rotations to improve contact and
constraint fit. It is optional and adds CPU work. Crucially, ARDY's command-line
generation **disables postprocessing for G1** with the code comment “does not
work well for this model.” The interactive demo still offers a toggle; neither
path establishes that G1 correction is suitable for StageZero. Compare it only
in an isolated probe after uncorrected baselines, with saved original arrays
and objective measures for foot slide, hand error, joint discontinuity, and
any new artifacts. Do not quietly enable it in the product.
[Pinned CLI](../vendor/ardy/scripts/generate.py),
[postprocess wrapper](../vendor/ardy/ardy/postprocess.py),
[bundled correction README](../vendor/ardy/MotionCorrection/README.md).

## Two actors from one checkpoint

The lowest-risk first prototype loads the checkpoint and encoder once and
keeps **two independent actor states**: each actor's prompt, normalized motion
history, root transform, target list, generated frames, and cancellation
version. Both actors share a 25 fps timeline. The isolated probe below now
compares sequential calls with a batch of two; a production scene still needs
two independent states and synchronized playback. Batched inference doubles
actor tensors and the denoiser's effective batch, so latency and VRAM cannot
be inferred from one-actor measurements. Although the pinned `Ardy`
constructor defaults to regular CFG, the
**actually loaded StageZero checkpoint uses
`AutoLatentClassifierFreeGuidedModelSeparated`**. It runs text-only,
constraint-only, and unconditional passes in one denoiser batch (effective
batch `3B`) and uses both entries of `(text, constraint)` guidance weights.
The measured loaded diffusion schedule has ten base steps. These are runtime
observations from the completed 108/108 ablation, and subsequent probes should
record the wrapper and step count rather than assume the constructor default.
[CFG implementation](../vendor/ardy/ardy/model/cfg.py),
[model construction](../vendor/ardy/ardy/model/ardy_model.py).

StageZero's earlier one-actor production baseline on the existing RTX 6000 Ada
was 0.205 s median and 0.213 s p95 Pod generation for **104 frames** (two
52-frame steps), with about 14.95 GiB peak allocated GPU memory. The following
isolated probe instead generated **one 52-frame horizon** per call from
52-frame histories, with one actor shifted 2 m in world X. These timings are
therefore comparable only within the probe, not directly to the 104-frame
production baseline. [StageZero measured review](../review/DIRECTING-RESULTS.md),
[probe report](../review/motion-interaction-probe.json).

| Paired seed | Two sequential calls, total | One two-actor batch, total | Batch peak allocated | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 0 | 0.606 s | 0.159 s | 14.955 GiB | First sequential call took 0.486 s, including 0.283 s text encoding; this is a cold/order-affected comparison. |
| 1 | 0.248 s | 0.154 s | 14.955 GiB | Batch ran first; subsequent sequential calls were warm. |

All 10 model calls succeeded, including both two-actor batches. The comparable
warm pair suggests a 1.61x throughput gain for these two short prompts, but
two seeds are insufficient for a stable latency or memory capacity claim. The
peak allocated difference from the one-actor calls was only about 0.006 GiB
in this report (14.949 vs 14.955 GiB); that does not predict larger batches,
longer horizons, or more complex conditions. Reserved memory was 15.010 GiB
in both modes, and checkpoint load took 72.9 s outside the timed calls. The
batch generated separate motion arrays and kept each actor's prompt in its
own sample. Prompt fidelity, partner awareness, inter-actor separation, and
joint playback were not scored. The measured history-to-output mean joint
jump was about 0.034–0.035 m for these cases, but that scalar cannot establish
smooth acting or safe physical contact. Floor penetration and foot slide were
not measured in this probe; they remain required checks before scene use.

The same probe tested a final-frame `Root2DConstraintSet` target 0.8 m in
world Z from the first actor's prior root. At model-window frame 103, the
unconstrained endpoint missed by **0.648 m and 1.090 m** for seeds 0 and 1;
the conditioned endpoint missed by **0.0068 m and 0.0102 m**. This is strong
evidence that the exposed root condition controls this simple endpoint on
these two seeds. It does not establish stopping, heading, smooth travel, foot
contact, or accuracy for other targets. Each conditioned call took about
0.122–0.131 s including text encoding and condition construction; seed 0's
first condition construction was 0.010 s, while the warmed seed 1 build was
0.0007 s. The conditions had two nonzero channels. [Case-level results](../review/motion-interaction-probe.json).

## Actionable interaction ladder

Use a single deterministic scene clock and a world-coordinate target plan.
Keep all targets in meters with +Y up. For each experiment save prompts,
constraint set and mask coverage, seed, checkpoint/commit, actor histories,
generated arrays, wall/GPU metrics, and target-error series. Run several seeds
because a single plausible clip is not evidence of reliable control.

1. **Approach marker (one actor):** the probe established a single final
   planar root target to within 1.1 cm on two seeds. Next add two or three
   `Root2DConstraintSet` waypoints within the visible window, then replan on
   the following 52-frame horizon. Measure target error, travel smoothness,
   floor penetration, foot slide, and post-target drift. A close endpoint
   alone is not a convincing approach-and-stop clip.
2. **Stop, face, and wave to a partner (two actors):** keep two independent
   histories on the shared 25 fps scene clock. The completed probe confirms a
   batch of two samples fits and runs on the current Pod for one horizon.
   Choose separated stop positions and reciprocal headings in one world
   frame, condition both actor roots at a common frame, then cue a wave after
   both are stationary. Measure position and heading error, body separation,
   residual walking, seam continuity, and visual legibility in synchronized
   playback. Add a planner separation rule: ARDY receives no live partner
   geometry.
3. **Reach to target (one actor):** start from a valid FK pose and make a hand
   keyframe at a reachable position relative to a fixed root. Compare sparse
   versus short interval constraints and position versus position+orientation.
   Measure hand point error at the requested frame, wrist orientation error,
   shoulder/elbow plausibility, torso movement, and local foot slide. Fail
   candidates that reach by penetrating the torso or shifting the planted feet.
4. **Prop reach, then shared hand contact / handoff (two actors):** first put a
   fixed prop at a known world transform and measure whether a constrained
   hand reaches its grip point while the body remains plausible. Then choose
   a meeting point and schedule a giver hand arrival, a short overlap, then
   receiver hand arrival
   on the same timeline. Derive hand targets from the intended prop grip
   offsets, not necessarily the prop center. First hold the giver's accepted
   trajectory fixed while generating the receiver against its *world-space*
   contact point; then test symmetric preplanned targets in one batch. Measure
   hand-to-hand and hand-to-prop distances, contact duration, hand speed at
   transfer, body separation, and continuity across replans. A deterministic
   prop attachment state may switch owner at the accepted contact frame; that
   visual event does not make the generated motion physically interactive.
   Grasp force, collision response, and continuous prop-aware movement remain
   unsupported by the current model contract.

For the ladder, define acceptance thresholds before seeing samples (for
example, maximum root/hand errors and minimum separation appropriate to the
G1 rig and prop scale), then report distributions and failures. Record any
manual target editing. If paired contact repeatedly needs incompatible poses,
move to constrained IK or a dedicated interaction model rather than hiding
large errors with scene logic.

## Boundary between API, experiment, and training

| Goal | Status with this checkpoint | Next evidence needed |
| --- | --- | --- |
| One final planar root target | Measured on two seeds: 0.0068 m and 0.0102 m endpoint error with conditioning | More seeds, waypoint paths, heading, stopping, floor and slide checks |
| Facing direction, reachable hand keyframe | Supported conditioning API; quality unverified here | Controlled GPU probe and error measurements |
| Two G1 actors on one timeline | Measured independent batch of two for one 52-frame horizon; cross-actor consistency is planner work | More seeds, synchronized playback, separation and quality review |
| Stop near partner, reach, visual handoff | Experiment: translate a shared plan into two sets of single-actor constraints | Contact/separation/continuity metrics across seeds |
| Prop-aware collision, force, grip, object state influencing pose | No scene geometry or object-state input in released ARDY/StageZero inference path | External planner/IK/physics or a model trained with such input |
| Robust spontaneous social interaction from one prompt | Outside the single-actor checkpoint's demonstrated contract | Interaction-paired data and a dedicated joint model/training objective |

The last two rows follow from the published single-human conditioning contract
and the pinned inference tensors; they are an engineering inference, not a
claim that every possible scene can never be approximated with waypoints. The
parallel object-layer work is separate from this model change. The new root
waypoint API supplies a destination, but does not encode prop geometry or grip
state into ARDY.
[NVIDIA ARDY overview](https://research.nvidia.com/labs/sil/projects/ardy/).
