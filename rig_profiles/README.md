# Supported humanoid rig mappings

`retargeting.build_retargeter(asset, mapping=None, skeleton=...)` accepts a GLB
validated by `character_assets.inspect_glb`. It consumes ARDY G1 global positions
`(34,3)` and global rotations `(34,3,3)` and returns local/world matrices for every
original glTF node. A renderer must apply the local matrices to their original
nodes. Root travel is already included in the pelvis matrix; do not translate the
whole GLB a second time.

The built-in profiles recognize exact G1 or Mixamo names (bare Mixamo names,
`mixamorig:` and `mixamorig` prefixes). Recognition is conservative. Renaming or
unrecognized rigs need a mapping JSON; recognition alone does not certify pose
quality. Two synthetic fixtures test the mathematical contracts. They are not a
claim of visual validation against all Mixamo exports or character artwork.

## Custom mapping

Pass a JSON object, JSON text or UTF-8 JSON bytes. Node references may be exact,
unique names or integer indices. The twelve roles below are required. Optional
roles are `left_hand`, `right_hand` and `head`. Hand endpoints are strongly
recommended because they establish the actual forearm axes. Unmapped fingers, twist joints and
helpers retain their local rest transforms and follow their animated ancestors.
G1 has no independent head motion, so a mapped head receives torso rotation.

```json
{
  "schema_version": 1,
  "profile_name": "my_humanoid",
  "bones": {
    "pelvis": "Hips",
    "spine": "Spine",
    "left_upper_arm": "LeftArm",
    "left_forearm": "LeftForeArm",
    "right_upper_arm": "RightArm",
    "right_forearm": "RightForeArm",
    "left_thigh": "LeftUpLeg",
    "left_shin": "LeftLeg",
    "left_foot": "LeftFoot",
    "right_thigh": "RightUpLeg",
    "right_shin": "RightLeg",
    "right_foot": "RightFoot"
  }
}
```

The optional `source_to_target_basis` is a proper orthonormal 3x3 rotation in
row-major JSON order. It converts source world vectors to target world vectors.
Without it, the hip left/right axis and shoulder/pelvis up axis define the basis.
The optional positive `root_scale` overrides the default ratio of target to source
leg length, expressed in target scene units per source metre. It changes root
travel only, never target limb proportions or mesh size. GLB export transforms
must already establish the intended character size.

Optional `bone_axes` maps a limb role to a finite nonzero direction in that
joint's local bind coordinates, for example
`"bone_axes": {"left_forearm": [1, 0, 0], "right_forearm": [-1, 0, 0]}`.
For a mapped child endpoint, the geometric axis is authoritative and an explicit
axis must agree with it. A terminal forearm without a mapped hand otherwise uses
a straight-arm bind assumption, following the upper-arm direction. This is
reported in `retargeter.warnings`; a bent forearm bind requires a hand mapping or
an explicit axis. The fallback cannot infer invisible bone tails from skin weights.

## Calibration and limits

Source bind comes from identity-local FK on the pinned G1 neutral skeleton, with
its lowest neutral joint at floor height. It never uses the first recorded or
generated motion frame as bind. Target rest comes from the imported glTF node
transforms. Each upper/lower arm and leg has a rest-axis swing calibration from
the target bind segment to the source neutral segment. Source global rotation
deltas apply to that calibrated orientation, preserving axial rotation. A final
swing correction aligns the posed target axis to the actual source segment from
the global position data. The actual posed parent transform converts the desired
world orientation into a local rotation. This handles intermediate unmapped nodes
and differing local bone axes while retaining local translations and uniform
scales. G1 terminal axis joints aggregate its shoulder, hip, waist, ankle and wrist
chains for **orientation**. Segment positions use anatomical landmarks separately:
shoulder pitch and hip pitch are the shoulder/hip centers; elbow, knee,
ankle roll and wrist yaw provide the other endpoints. The shoulder/hip yaw
motors sit farther along each limb and must not be treated as the anatomical
joint center. Doing so shortens measured limbs, distorts limb directions, and
overestimates root travel. Pelvis/torso/foot orientation follows calibrated global
rotation deltas.

G1 neutral deliberately does **not** leave a T-pose target in a T-pose: G1's upper
arms point mostly down and its forearms point forward. For the synthetic T-pose
target, preserving bind offsets caused 90.000° upper-arm and 89.413° forearm
direction errors. Segment calibration removes those errors. Lowered/raised-arm
tests verify actual shoulder/elbow/hand positions on both naming schemes, while
the target's own segment lengths stay unchanged.

`retargeter.bind_pose()` separately returns the untouched imported glTF rest.
Bind correctness means those node transforms still reproduce the mesh's inverse
bind relationship; it does not mean two different skeleton neutral poses are the
same human posture. `neutral_source_pose()` returns the actual G1 calibration
pose; this has bent robot legs and forward forearms and is not a relaxed human
standing pose. The character lab uses a separate, FK-consistent standing test
pose without changing that production calibration reference.
Limb direction matching does not solve body contacts, balance or twist distribution;
those require a more specific solver and visual evaluation.

Requirements and current limits:

- A single connected humanoid pelvis hierarchy, with each required mapped node a
  skin joint and each child role descending from its expected parent role. Helpers
  may sit between those roles. Multiple skins sharing that hierarchy are allowed.
- Unique mapped nodes, nonzero limb lengths, and nondegenerate rest axes.
- Positive uniform scales and proper rotations in node transforms; reflections,
  shear and nonuniform scale are rejected with an explanatory error.
- Finite G1 poses with proper rotation matrices. Other motion skeletons need their
  own source mapping/calibration.
- Root motion transfers once, scaled by leg length; local target bone lengths
  remain intact. No IK, foot locking, contact correction, independent finger/face
  articulation, physical balance correction, or arbitrary quadruped rig support.

Use the imported rest pose and a diagnostic single-joint pose to visually verify
an asset before relying on generated motion. The generic layer preserves GLB
materials/skin data; its visual fidelity is the renderer's responsibility.
