# Actual model selection — 26 September 2026

Selected showcase: **Kimodo RP v1.1, martial combination, seed 6201**. The actual generated take has a readable guarded sequence, a high kick around 3.45 seconds and a return to guard by 7 seconds. It is the strongest complete action among these reviewed samples. This is a selection of a particular take, not a claim that Kimodo always outperforms Core.

## Same-prompt eight-second samples

Both models received exactly these prompts and seeds:

| Action | Seed | Exact prompt |
|---|---:|---|
| Hip-hop | 6200 | A person performs an energetic hip hop dance with deep knee bends, alternating side steps, quick torso twists and broad rhythmic arm swings. |
| Martial combination | 6201 | A martial artist performs a sharp combination: a left jab, a right cross, a front kick, then resets into a low guarded stance and repeats the combination. |

Core source files are `core_hiphop_8s.npz` and `core_martial_combo_8s.npz`: native Core27, 160 frames at 20 fps. Kimodo source files are `kimodo_hiphop_8s.npz` and `kimodo_martial_combo_8s.npz`: native SOMA77, 240 frames at 30 fps. Portable review fixtures are bundled under `grounded_assets/clips/` (Core and converted motion) and `grounded_assets/source-motion/` (unaltered native Kimodo plus its skeleton). Additional rejected trial archives are retained in `grounded_assets/rejected/`. Kimodo used 100 generation steps and its official post-processing, as recorded by the source metadata. Raw output was retained unchanged after that official generation path.

## Visual decisions

- **Kimodo martial selected:** punch/guard transitions, a clearly raised kicking leg and a recovery are visible within the take. Static images do not establish timing quality; final playback review remains necessary.
- **Core martial retained as comparison, not the showcase:** the final kick has a large silhouette, but the clip ends mid-kick at 7.95 seconds. It does not give the same complete recovery.
- **Neither hip-hop sample selected as an energetic-dance showcase:** Core has larger crossed-arm gestures but modest footwork. Kimodo has a small shuffle and torso/arm gestures with relatively quiet opening and ending poses. Neither warrants calling the requested energetic hip-hop successful.
- The earlier procedural/swing presentation remains rejected. These samples use the actual textured generated human, not that rejected character or an authored swing-pose substitution.

[Hip-hop comparison](kimodo-vs-core-hiphop-8s.png) · [Martial comparison](kimodo-vs-core-martial-8s.png)

The sheets use exact CPU linear skinning of the generated mesh and its actual texture, with simple diagnostic lighting. Views are centered on each frame's pelvis to compare articulation; this camera framing does not modify the motion archive. Hip-hop shows matching times 1, 3, 4.65, 6 and 7.5 seconds. Martial shows action-relevant moments: Core at 1, 3.8, 5.75, 7.1 and 7.95 seconds; Kimodo at 1, 2.5, 3.25, 3.45 and 7 seconds. Rows are labeled rather than implying synchronized action beats.

## Explicit conversion and what its checks mean

`grounded_soma.py` writes separate `*_core27.npz` copies. Its named SOMA-to-Core27-layout adapter retains native SOMA proportions and endpoints; it does **not** force canonical Core bone lengths. It samples uniform 20 fps timestamps with local SO(3) interpolation, native-hierarchy forward kinematics and linear root interpolation. One extra Core-layout spine point is an explicitly derived midpoint; HandEnd maps to the native Middle2 joint. Root path and units are retained. The original 77-joint archives preserve detailed fingers, eyes and jaw that the reduced layout omits.

For the eight-second hip-hop and martial samples, the source FK consistency errors were respectively **1.97e-7 m** and **2.07e-7 m**. These numbers verify that the supplied native rotations, rest skeleton and positions use a consistent convention. They do **not** measure realistic anatomy, skin quality, physical contact, prompt fulfillment or perceptual animation quality.

Generated-mesh playback fits an approximate Human17 rig with fixed mesh proportions. Solo comparison sheets use the direct directional retarget, without arm target IK. A disclosed, constant floor translation is estimated per clip; it does not remove jumps or edit individual poses. Separate paired-dance playback can use per-frame native wrist-target IK and frozen floor offsets. That paired setting is not evidence of a secure grasp.

## Remaining limitations

- The actual generated mesh has thin rigid fingers, open hands and no facial animation. Martial motion therefore does not render clenched fists, despite readable body action.
- The anatomical rig is fitted, with compressed spine articulation and approximate shoulder skinning. Cloth, muscle and contact deformation are not physically simulated.
- Small sole-height errors and contact drift remain. Paired dance has intermittent hand proximity, not a sustained reliable grip.
- Static contact sheets support pose and deformation inspection; they do not certify temporal smoothness or replace the final browser playback capture.

Raw Kimodo SHA-256: hip-hop `7f4436e78988f7b5004cc447509a8e7d744e16e7e2471acbe74497ff149e6da4`; martial `7040da737abb041ebf04bcd1d8173284949b36efe0ac66b253976860f1b0482c`. Adapter tests recheck raw hashes and native endpoints at coincident timestamps. The combined focused retarget/adapter suite passes 16 tests (including frozen-prefix/isolated-frame clearance and native-world wrist-coordinate regressions).


## Bounded stunt trial and endpoint correction

Four further actual Kimodo eight-second samples were reviewed. Each raw take contains 240 native SOMA77 frames at 30 fps; separate named conversions contain 160 frames at 20 fps. Timings are generation only, not transfer, conversion, fitting or browser-ready latency.

| Take | Seed | Generation seconds | Exact prompt | Visual decision |
|---|---:|---:|---|---|
| Cartwheel | 6301 | 3.453 | A trained acrobat takes two quick steps, performs a cartwheel, lands on both feet and stands balanced. | Native inversion succeeds, but the fitted hands do not establish convincing palm support. Retain as an experiment; reject as a contact-quality showcase. |
| Spinning roundhouse | 6302 | 3.345 | A martial artist performs a fast spinning roundhouse kick, lands solidly, then punches twice and returns to guard. | Readable large spinning high kick and recovery. Useful additional action take; rigid open hands remain. |
| Breakdance sweep | 6303 | 3.375 | A dancer performs energetic breakdance footwork, crouches into a low floor sweep, then rises with arms spread. | No actual floor sweep: standing and crouching gestures. Reject prompt fulfillment. |
| Shoulder roll | 6304 | 3.315 | A stunt performer ducks, performs a forward shoulder roll, gets back onto their feet and runs two steps. | No forward shoulder roll: shuffle, turn and running movement. Reject prompt fulfillment. |

[Cartwheel baseline](kimodo-cartwheel-8s-review.png) · [Spinning roundhouse](kimodo-spinning_roundhouse-8s-review.png) · [Breakdance failure](kimodo-breakdance_sweep-8s-review.png) · [Shoulder-roll failure](kimodo-shoulder_roll-8s-review.png). All bad attempts remain available.

The cartwheel exposed a real retargeting defect: applying the canonical pelvis-height offset to native world wrist targets pushed hand-support trajectories down by 14.69 cm. The new optional `wrist_target_space='native_world'` corrects that coordinate convention. It retains actual per-frame native wrist trajectories plus the disclosed fixed clip floor offset, using standard two-bone arms. Existing paired settings are unchanged.

A trial that also targeted full native world ankles was rejected: the generated mesh's different leg proportions left 87.8% of cartwheel ankle targets unreachable, with a 34.45 cm maximum miss. Less floor penetration alone did not make that trial acceptable.

The replacement opt-in `preserve_feet=True` is specifically **vertical boot clearance**, not preservation of exact native ankle coordinates. It keeps the existing fitted ankle x/z, pelvis, foot orientation and fixed bone lengths. Only actual penetrating foot mesh near native ground contact lifts vertically through two-bone leg IK. Eligibility fades continuously between native toe heights 15 and 30 cm; airborne and nonpenetrating frames stay unchanged. There is no downward snap, foot lock, per-frame root translation, future-dependent correction or contact invention. A caller must supply `floor_y`; continuation must retain the initial `floor_offsets` rather than recalibrating on the expanded clip.

[Exact skinned comparison](cartwheel-clearance-only-review.png) shows native-world wrist fitting above and the additional boot-clearance mode below at frames 20, 53, 59, 64, 68 and 145. Landing knees remain plausible and the boots no longer pass deeply through the floor. The cartwheel's maximum whole-mesh penetration is now 7.1 mm (hands), down from about 22.6 cm in the original directional fit. The maximum joint step remains 51.5 cm at 20 fps, arising during the fast native inversion; it is not evidence of a slow or smooth gesture. The fully inverted frame still shows open rigid hands hovering visibly above the floor, so the corrected take is **not** certified as a supported handstand.

The independent [clearance audit](foot-clearance-independent-audit.json) checks four actual clips. On the 876-frame live solo clip, the previous 8.22 cm boot penetration is removed; maximum frame movement increases only 2.41 mm, without worsening the measured continuation seam. Fixed-offset prefixes and isolated-frame outputs are bit-identical, pelvis is unchanged, airborne leg transforms are unchanged, ankle x/z errors are below 1e-15 m, and all clearance targets are reachable. No new stance hover above 5 cm was detected by the stated native low/slow-foot proxy. This validates the bounded clearance mechanism, not foot planting: sliding, approximate anatomy and rigid hands remain limitations.


A later localized continuity check strengthens the cartwheel rejection: at frame 71, the clearance correction increases the left-knee step from 1.89 to 10.10 cm, although the global maximum joint step is unchanged. A nearly extended leg amplifies a small ankle lift, and the native knee plane turns during the recovery. [Frames 46–47 and 69–72](cartwheel-clearance-knees-review.png) expose the resulting knee adjustment. This take is excluded from the default showcase; no further corrective experiments were accepted. On the live Core clip, the largest added knee-step magnitude is only 8.76 mm. The full-native-ankle trial remains rejected and is not the implemented clearance mode.
