# Native ARDY pose-goal experiment

2026-09-26 · existing RTX 6000 Ada Pod · pinned `ARDY-G1-RP-25FPS-Horizon52`

The isolated probe completed all 10 paired held-out cases (seeds 101–105 for
each action). Every generated clip came directly from ARDY inference. No
generated positions or rotations were edited. Each pair shared the same
original walking prefix, first 52-frame horizon, and second-horizon random
state. The only changed input for the second horizon was ARDY's native pose
condition tensor and mask.

| Action | Unconditioned proxy | Pose conditioned proxy | Native constraint |
| --- | ---: | ---: | --- |
| Both hands above shoulders by >15 cm for 5 continuous frames | 2/5 | 5/5 | Three coherent hand/hip keyframes from a successful generated overhead clip; 105 masked channels |
| Pelvis drops ≥15 cm relative to walking prefix | 1/5 | 5/5 | Three coherent full-body keyframes from a successful generated squat clip; 312 masked channels |

All 20 clips passed finite pose, mean joint seam ≤15 cm, and toe penetration
≤5 cm guards. The actual maximum two-horizon seam was under 5 mm mean joint
displacement; maximum toe penetration was under 5 mm. Mean second-horizon
generation plus decoding was about 0.07 seconds for either variant on this
Pod; model loading took 67.25 seconds and was excluded from that timing.

Visual review shows a clearer two-hand raise and a deeper squat in the
conditioned examples: [overhead](../motion-goal-overhead.png) and
[squat](../motion-goal-squat.png). The raised hands are near the helmet rather
than fully straight overhead. The squat has forward lean and hands close
together. These are reference-guided G1 model outputs, not evidence that
arbitrary acting prompts or physical contacts are solved. Toe sliding is a
geometric proxy; it does not establish foot contact or balance. The squat
proxy improved while its mean toe-sliding proxy increased slightly in these
cases (roughly 0.03–0.08 to 0.04–0.09 m/s).

The source reference keyframes came from the original ablation's successful
seed-33 `h4cfg2` overhead and `h12cfg4` squat outputs. The probe aligned
world XZ translation only. The reusable
[`motion_action_goals.py`](../../motion_action_goals.py) also aligns heading,
with CPU tests for +90° yaw and native masks; rotated-history GPU behavior
requires separate real inference verification before broad use.

Full cases, keyframe locations, translated goal coordinates, timing, and
quality measurements: [`report.json`](report.json). Reproducible probe:
[`action_constraint_probe.py`](../../experiments/action_constraint_probe.py).
