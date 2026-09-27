# Core terrain contact v2: bounded offline assessment

This is an **experimental presentation adapter**, not part of ordinary Core generation or rendering. Native Core27 feature tensors and history are unchanged. The candidate infers stance from native toe speed and toe height relative to pelvis, holds an anchor within each observed stance, and solves each leg from its native knee bend plane. Root XZ, root orientation, and every non-leg global rotation remain unchanged. Root Y corrections are chosen from feasible values and only the corrections are smoothed.

The acceptance limits were set before checking the local stair cases: foot relative to root 0.12 m, residual root Y 0.08 m, root XZ 0.06 m, local leg rotation 35°, root correction rate 0.04 m/frame, foot correction rate 0.08 m/frame, stance gap/slip 0.025 m, sole and swept penetration 0.01 m, and supported footprint height spread 0.035 m. A failing case is rejected; these limits were not enlarged.

| Local-floor native case | Native minimum toe clearance | Candidate maximum root Y | Candidate maximum foot relative correction | Candidate maximum sole penetration | Candidate maximum local leg rotation | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sparse seed 33 | -0.078 m | 0.070 m | 0.105 m | 0.100 m | 47.6° | Reject |
| sparse seed 11 | -0.180 m | 0.065 m | 0.102 m | 0.033 m | 45.1° | Reject |
| foot-hint seed 11 | -0.111 m | 0.075 m | 0.116 m | 0.148 m | 49.0° | Reject |

All three also failed the correction-rate, footprint-spread, and swept-penetration gates. The foot-hint case had 12 frames with no root-Y solution inside the budget and 16 unresolved leg targets. These sole results are from a four-point Core foot proxy, while the native metric measures the toe landmark, so their penetration numbers are not directly comparable. The proxy tests actual scene support at each sampled point but is not a full skinned mesh or a physical collision model. The candidate has no streaming anchor state and has not passed visual review. **Do not promote it to the runtime.**

Reproduce the three-case assessment from the project root:

```sh
PYTHONPATH=.:vendor/ardy /Users/jakob/Desktop/Shellhacks/.venv/bin/python experiments/core_terrain_contact_v2.py \
  --scene review/spatial-v2/terrain-native-localfloor/scene.json \
  --input review/spatial-v2/terrain-native-localfloor/height_sparse__seed33.npz \
          review/spatial-v2/terrain-native-localfloor/height_sparse__seed11.npz \
          review/spatial-v2/terrain-native-localfloor/height_foot__seed11.npz \
  --output /private/tmp/core-terrain-contact-v2-trial
```

The command writes one JSON gate report and one rejected candidate NPZ per case. Focused unit checks:

```sh
PYTHONPATH=.:vendor/ardy /Users/jakob/Desktop/Shellhacks/.venv/bin/python -m unittest test_core_terrain_contact_v2 -q
```
