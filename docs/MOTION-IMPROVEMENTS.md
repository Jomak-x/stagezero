# Motion model improvements and measured limits

Status, 2026-09-26: **directed-v2 is deployed on the existing Pod** and the
unchanged live inference endpoint passes the verification below. The studio UI
and object-generation work were not modified for this delivery.

## Live verification

- **61 regression tests pass**, including official CPU constraint construction,
  yaw/translation alignment, cancellation, options, and candidate quality checks.
- **20/20 production generation calls passed** (19 named cases and one exact
  deterministic repeat). Default overhead and squat goals each passed their
  geometric cue in **5/5 seeds**. Wave and stop each passed 2/2 checks. These
  small samples establish bounded behavior, not arbitrary acting reliability.
- A final untouched-seed check (201–210) passed **10/10 overhead and 10/10 squat**
  geometric cues on the live endpoint: [fresh-seed evidence](../review/motion-fresh-seeds.json).
  These use the same walking prefix; different scene histories remain unmeasured.
- Four isolated endpoint waypoint checks reached frame-51/frame-103 targets
  within **0.0037–0.0303 m**; the paired free-motion errors were 1.00–2.81 m.
- Both goal actions passed with a history transformed through ARDY's official
  representation by **90° yaw and [2, -1] m translation**.
- Malformed options and unsupported explicit goals return 400; cancellation
  returns 409. A saved three-take project retained the original 104-frame prefix,
  all generated arrays bit-for-bit, and reference-goal metadata after reload.
- Across 14 default production requests, GPU generation median/p95 was
  **0.211/0.407 s**, and SSH round-trip median/p95 **0.495/0.695 s**, including
  the first request's warmup. Loading the deployed model took 69.83 s.

Evidence: [production verification](../review/motion-live-validation.json),
[candidate goal verification](../review/motion-backend-goals.json),
[waypoint measurements](../review/motion-backend-targets.json), and
[rotated-history verification](../review/motion-backend-rotated.json).
The optional waypoint API is ready for scene integration; the current client
does not yet send object geometry, grasp state, or paired actor conditions.

Rendered live outputs: [overhead](../review/motion-live-overhead.png) and
[squat](../review/motion-live-squat.png). A raised-hand geometric cue does not
mean fully extended arms; G1 overhead samples can remain asymmetric and the
squat can lean forward. The separate Core human rig is more promising for
full-body human scenes, with different integration requirements.

The local review project is `.runtime/projects/StageZero-movement-v2.stagezero.npz`.
The previous Pod backend remains at
`/workspace/stagezero/motion-backups/before-directed-v2/pod_backend.py` for rollback.
Deployments must include the four motion modules and `assets/motion-goals.npz`
alongside `pod_backend.py`; updating only the backend file is insufficient.

## What changed

StageZero continues to use NVIDIA's pinned `ARDY-G1-RP-25FPS-Horizon52`
checkpoint on the existing RTX 6000 Ada Pod. No weights were fine-tuned or
retrained. We changed inference control: the backend can use a shorter recent
motion history, adjust text guidance, pass native pose goals into ARDY, and
select an actual model output using conservative geometry and continuity
checks. Selection never edits or fabricates poses. The existing 104-frame
request still contains two 52-frame model steps at 25 fps.

The proposed `directed-v2` automatic policy uses a **responsive** profile
(four frames of initial and carry history, text CFG 2) for ordinary prompts.
An explicit both-arms-overhead instruction uses one responsive sample with
three coherent hand/hip reference keyframes in the second horizon. An explicit
squat instruction uses one **expressive** sample (12 frames of initial and
carry history, text CFG 4) with three coherent full-body keyframes in the
second horizon. These poses come from successful prior G1 model generations in
[assets/motion-goals.npz](../assets/motion-goals.npz). They are aligned to the
current actor's world X/Z position and yaw, then encoded as ARDY's soft native
conditions. The first horizon remains free. The asset must ship alongside
[motion_action_goals.py](../motion_action_goals.py) and the backend.

For a recognized stop, automatic policy uses one expressive sample; a
responsive stop regressed on held-out seeds. For other or ambiguous prompts,
automatic policy uses one responsive sample without a pose goal. Setting
`generation_options.pose_goal` to `false` restores the earlier two-candidate
responsive/expressive policy for automatic overhead and squat prompts. An
explicit `profile` turns automatic pose goals off unless `pose_goal: true` is
also requested. An explicit pose goal is accepted only for an unambiguous
overhead raise or squat; other prompts receive a validation error. Supplying
`motion_target` disables automatic pose goals, and requesting both an explicit
pose goal and root waypoint receives a validation error.

The selector favors a sample that reaches the recognized geometric cue,
subject to finite poses, valid joint rotations, no more than 5 cm of measured
floor penetration, no more than 15 cm mean joint displacement at the preceding
history or internal 52-frame seam, and no frame-to-frame mean joint step over
15 cm anywhere in the clip. All 248 saved G1 clips used to audit this last
guard passed it; passing the guard is not proof of natural motion. See
[policy](../motion_policy.py), [quality analysis](../motion_quality.py), and
[backend](../pod_backend.py).

The backend also accepts an optional planar root waypoint through ARDY's
official conditioning API. It is a model condition, not a position applied to
the returned pose. The current request shape is shown below; `history` is a
placeholder for the existing numeric G1 array:

```json
{
  "prompt": "A person walks to the marker and stops there.",
  "history": "existing 4–52 frame G1 history array",
  "motion_target": {"position_xz": [0.4, 2.7], "frame": 103},
  "generation_options": {"seed": 101}
}
```

`motion_target` needs prior actor history; its world-space X/Z point must be
within 3 m of the prior root. Only frame 51 or 103 of the 104-frame result is
accepted. A frame-103 target is introduced in the second 52-frame step, so the
first step does not plan ahead to it. The target contract, including strict
validation and model-mask construction, is in [motion_constraints.py](../motion_constraints.py).
The response metadata records the chosen seed, base seed, each candidate's
profile, seed, history length, guidance weight and step timings, the selector's
quality and action-proxy assessments, applied pose-goal details and asset hash
when relevant, the submitted root target, and its measured endpoint error. A
caller may explicitly select `legacy`, `responsive`, or
`expressive` and request one to three candidates through `generation_options`;
these options allow reproducible comparison. Automatic policy applies when no
profile is supplied. Failed quality guards reject the new motion so an existing
take can be retained.

## Controlled model results

We continued the same saved 104-frame walking prefix for every test, with
four prompts: both arms overhead, right-hand wave, squat, and stop. Discovery
used three seeds and nine inference settings: **108/108 model calls completed**.
Its strongest overhead and squat settings were then tested on ten new seeds
(101–110): **120/120 held-out model calls completed**. All cases generated
104 frames. The full plans, per-case poses and metrics are in the
[discovery summary](../review/motion-ablation-summary.json) and
[held-out summary](../review/motion-validation-summary.json).

The following table reports the earlier **text-only** policy, available with
`pose_goal: false`. It does not measure the new default reference-assisted
`directed-v2` behavior. It uses the actual selector's cue checks for 40 held-out decisions
(ten seeds per action). The hand check requires five *consecutive* frames with
the relevant hand at least 15 cm above its same-side shoulder; the squat check
requires a 15 cm pelvis drop; the stop check requires mean planar root speed
at most 0.15 m/s in the final 25 frames. These are useful geometric proxies,
not a measure of acting quality or physical correctness.

| Prompt | Legacy 52-frame history, text CFG 2 | Earlier text-only policy | Samples per request |
| --- | ---: | ---: | ---: |
| Both arms overhead | 0/10 | **5/10** | 2 |
| Right-hand wave | 10/10 | 10/10 | 1 |
| Squat | 1/10 | **3/10** | 2 |
| Stop and stand still | 10/10 | 10/10 | 1 |

That text-only selector chose responsive or expressive for overhead 3/7 times and squat
5/5 times. All 40 selected candidates passed its quality guards. The discovery
three-seed results were more optimistic for squat (3/3 with expressive); the
held-out 3/10 is the more relevant estimate. Shorter history improves the
chance of escaping the previous walk, but text-only overhead and squat remain
unreliable for arbitrary seeds. The original held-out summary counts any five
qualifying hand frames, while the selector demands consecutive frames; the
[policy validation](../review/motion-policy-validation.json) recomputes all 40
outcomes with the actual selector and reports both definitions.

### Native pose-goal experiment for the new default

A separate paired probe held each seed's original walking prefix, first
52-frame horizon, and second-horizon random state fixed. Only the native ARDY
condition mask and tensor changed in the second horizon. The keyframes came
from successful seed-33 G1 generations, then were translated to each actor's
world X/Z position. The reusable backend also aligns yaw, which needs further
real GPU testing with rotated histories. All ten paired cases across five
held-out seeds (101–105) completed; all 20 resulting clips passed the probe's
finite-pose, seam, and floor-depth guards. The saved
[probe report](../review/motion-constraint-v1/RESULTS.md) and
[case data](../review/motion-constraint-v1/report.json) give the exact
conditions and measurements.

| Action proxy | Same-seed text-only second horizon | Native pose-conditioned second horizon |
| --- | ---: | ---: |
| Both hands >15 cm above shoulders for five continuous frames | 2/5 | **5/5** |
| Pelvis drops at least 15 cm from the walking prefix | 1/5 | **5/5** |

The overhead condition used three coherent hand/hip keyframes (105 masked
channels); squat used three full-body keyframes (312 masked channels). The
largest two-horizon mean joint seam and toe penetration in the probe were each
under 5 mm. Mean second-horizon generation and decoding took about 0.07 s
for either variant, excluding the first horizon, model loading, transport,
and backend selection. The probe used only five seeds and X/Z alignment; it
does not establish a general success rate for the new default in the live
endpoint. [Overhead goal screenshot](../review/motion-goal-overhead.png) and
[squat goal screenshot](../review/motion-goal-squat.png) show saved examples.
The hands rise near the helmet rather than fully straight overhead; the squat
leans forward and brings the hands together. Those details matter when judging
whether an acting scene looks right, beyond its geometric proxy.

The saved isolated text-only run timings imply mean generation cost of 0.287 s for two
overhead candidates versus 0.171 s legacy, and 0.315 s for two squat candidates
versus 0.145 s legacy. Wave and stop single-candidate means were roughly
0.143 s versus 0.147 s and 0.142 s versus 0.145 s. These are case-level model
timings, not end-to-end UI latency; they exclude checkpoint loading, network,
queueing, and the final selector. The first case includes warmup. The
checkpoint itself took about 68–74 s to load in the isolated runs.

Visual evidence uses the exact saved arrays, including the original walking
prefix: [legacy overhead](../review/motion-overhead-baseline.png),
[improved overhead on a discovery seed](../review/motion-overhead-improved.png),
[held-out overhead example](../review/motion-heldout-overhead.png), and
[improved squat on a discovery seed](../review/motion-squat-improved.png).
These screenshots show particular frames, not a success guarantee across
seeds or proof that the full animation reads naturally. Measured floor and
foot-slide values are proxies; they do not establish foot contact, balance,
correct squat form, hand oscillation for a wave, or contact with props.

### Separate human-rig exploration

An isolated [ARDY Core probe](../review/motion-core-probe.json) generated
24/24 80-frame human-rig clips at 20 fps for wave, overhead, squat, and
startled-turn prompts (three seeds and two guidance strengths). The
[Core model note](MOTION-MODEL-OPTIONS.md) records its measurements and
integration limits; [local Core review viewer](http://127.0.0.1:2340/) and
the [overhead](../review/motion-core-overhead.png),
[squat](../review/motion-core-squat.png),
[wave](../review/motion-core-wave.png), and
[turn](../review/motion-core-turn.png) captures allow visual comparison.
Core uses a different 27-joint, 330-feature rig and a 40-frame horizon. This
probe is separate from the G1 backend policy and is not a drop-in model switch
for the current single-actor demo.

## Reproducing the evaluation

The GitHub regression job initializes the pinned ARDY submodule and installs
CPU-only torch 2.8.0 plus NumPy, requests, and einops; no model weights, gated
data, GPU, or Pod credentials are needed for these tests. With the full local
environment already installed, run:

```sh
.venv/bin/python -m unittest test_live_motion test_directing test_director_edges \
  test_motion_quality test_motion_policy test_motion_constraints \
  test_motion_action_goals test_backend_policy
```

The private source take and raw per-case arrays are under `.runtime/` in the
local workspace and were not included as public source assets. On a GPU host
with the pinned model cached and the same runtime, the runner and scoring
commands are:

```sh
.venv/bin/python experiments/motion_ablation.py \
  --history-project .runtime/projects/StageZero-two-endings-20260926-015459-6a268b.stagezero.npz \
  --output .runtime/motion-research/ablation-v1
.venv/bin/python experiments/motion_ablation.py \
  --history-project .runtime/projects/StageZero-two-endings-20260926-015459-6a268b.stagezero.npz \
  --output .runtime/motion-research/validation-v1 \
  --seeds 101,102,103,104,105,106,107,108,109,110 \
  --configs h52cfg2,h4cfg2,h12cfg4
.venv/bin/python experiments/summarize_motion_ablation.py \
  --input .runtime/motion-research/validation-v1 \
  --history-project .runtime/projects/StageZero-two-endings-20260926-015459-6a268b.stagezero.npz \
  --output review/motion-validation-summary.json
.venv/bin/python experiments/evaluate_motion_policy.py \
  --input .runtime/motion-research/validation-v1 \
  --summary review/motion-validation-summary.json \
  --history-project .runtime/projects/StageZero-two-endings-20260926-015459-6a268b.stagezero.npz \
  --output review/motion-policy-validation.json
.venv/bin/python experiments/action_constraint_probe.py \
  --source-project .runtime/projects/StageZero-two-endings-20260926-015459-6a268b.stagezero.npz \
  --exemplars .runtime/motion-research/ablation-v1 \
  --output .runtime/motion-research/constraint-v1
```

The [interaction research note](MOTION-INTERACTIONS-RESEARCH.md) records a
separate small two-actor and planar-root probe, including its scene-use-case
ladder and limitations. The official RunPod CLI, `runpodctl` 2.14.0, is
installed locally. API-key configuration is pending; access to the existing
Pod currently works through its existing SSH credentials. The other
"onboarding" Pod was not needed for this work. No new resources were
provisioned.
