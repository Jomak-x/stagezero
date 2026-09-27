# Native Core terrain conditioning trial

No raw native candidate passed foot support verification. Eighteen candidates were generated on existing pod1: three variants × three seeds in absolute scene coordinates, then the same matrix with a rigid floor coordinate frame per horizon. No services were restarted and no new pods were created.

The authentic corrected bridge-approach stair flight contains nine treads with 0.097826 m rise and 0.275510 m run. Start support Y=2.030464 m and end support Y=2.900662 m. Each candidate contains 160 generated frames at 20 fps, after a common 40-frame standing prefix. The prompt, seeds 11/22/33, ten native denoising steps and CFG (2,2) stay fixed.

The native root_y_pos feature masks only root Y. Root2D without heading masks only XZ. Sparse toe-position constraints add position feature channels and no ancestry rotations; their FK toe positions do not necessarily match the hints.

## Results

| Conditioning | Absolute-coordinate worst toe clearance | Local-floor worst toe clearance | Local-floor toe samples below -5cm |
|---|---:|---:|---:|
| Sparse XYZ | -0.501 to -0.476 m | -0.180 to -0.078 m | 10–20% |
| Dense XYZ | -0.425 to -0.415 m | -0.176 to -0.099 m | 24–40% |
| Sparse XYZ + toe positions | -0.524 to -0.508 m | -0.164 to -0.111 m | 18–23% |

Absolute-coordinate standing prefix ended at pelvis Y=3.473522 m for a target of 2.980464 m. The rigid local-floor prefix ended at 2.965860 m. Native Core does not vertically recenter its history; this large absolute-height mismatch is consistent with the known representation limitation. This experiment establishes improvement from changing coordinate origins, not reliable terrain traversal.

`fk-contact-sheet.png` visually confirms the absolute-height feet floating initially then intersecting stairs. Local-floor candidates show visible walking and better height tracking, but toes still pass through tread edges. Core foot joints are points, so these metrics do not establish skinned sole clearance, full-body clearance, dynamics, or visual gait quality.

## Coordinate handling

For each 40-frame horizon, one constant floor origin is subtracted from all history root Y and ground-relative local-joint Y features and from all conditioning coordinates. The same constant is added to returned native features before official FK decoding. Rotations, velocities, contacts and relative joint poses are preserved. Five CPU tests verify the feature mask, history mapping, default-empty conditions, batch offsets, exact rigid FK translation, inverse transform pair, and unchanged input arrays. Native output positions independently decoded from the saved restored features agree within 1 micrometre.

## Review archives

- `height_sparse__seed33.core.npz`: best sparse-height candidate by toe clearance, 160 frames; raw native support still fails.
- `height_sparse__seed11.core.npz`: representative worse sparse-height candidate, 160 frames.
- The corresponding `__with_prefix.core.npz` files preserve the full 40-frame prefix plus 160 generated frames.
- Every archive roundtrips positions, rotations and native features bit-for-bit through `CoreStudioSession`.
- `terrain_navigation_version=1` marks explicitly requested terrain playback; `raw_native_passed=false` preserves the failed physical result.
- `scene.json` and `studio_core.source_scene` preserve the complete source scene. Playback snapshots omit only the unsupported `handle_height` metadata property on the distant shrine door so the existing scene schema can load them; stair geometry is unchanged.
- All absolute-height trials remain in sibling `../terrain-native-v2/`; all local-floor trials and both cached text embeddings remain here. `source/` snapshots the experiment and native transform code, and `core-motion-stats/` holds the public model normalization data used to decode the prefix.

Runtime ordinary generation was not edited by this worker. The height constraint and coordinate transform are reusable opt-in mechanisms, not a terrain-success claim. Any subsequent renderer contact adjustments must be reported separately from these saved native outputs.
