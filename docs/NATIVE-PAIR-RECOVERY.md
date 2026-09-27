# Native paired-motion recovery

The user rejected both the independent Core duet and the later retargeted InterGen scuffle. Neither is an accepted animation milestone. Their evidence is preserved.

## What went wrong

The earlier handshake and embrace preview rendered InterGen's original 22 joints directly at 30 fps, over a full 210-frame sample. The later studio path instead generated 180 frames, converted them to Core27 at 20 fps, and fitted them again to an approximate 17-bone skin rig. That is not the same clip or the same representation.

A CPU audit of the original handshake42 found substantial pose changes in both conversions. The Core bridge moved wrists by about 8 cm on average and changed the left elbow angle by about 38.5 degrees. The subsequent character fit moved elbows another approximately 20 cm and changed elbow angles by approximately 85–92 degrees. Its arm chain is 65.7 cm versus the Core chain's 52.8 cm. Preserving a wrist target with the wrong arm proportions forced excessive elbow bending. The approximate skin and torso mapping compounded the visible defect.

The earlier tests covered archive, timing and workflow mechanics. They were not sufficient animation acceptance, and the hand-distance measurements hid damage elsewhere in the body.

## Recovery and experiments

1. Restore the exact original handshake42 and embrace37. Keep their raw source arrays and original hashes.
2. Render native30 motion directly with a solid articulated mannequin and a raw joint overlay. This isolates motion quality from skinning.
3. Fit a properly authored humanoid skin directly to native22 motion, bypassing both Core and the synthetic skin rig. Review deformation on the full scene before considering integration.
4. Audit the native rotation features as an optional way to preserve axial orientation. Position-only motion does not specify finger articulation.
5. Generate bounded fresh full-length paired scenes on the existing Pod: handshake, embrace, waltz, dodge, high-five and helping someone stand. Keep failures and judge full sequences.

Original raw archive copies and reproducible geometric measurements are in `review/two-character/native-recovery/`. The restored original viewer runs on localhost:2345 behind the existing private Tailscale port 2343 route. That route previously returned 502 because the viewer had stopped; no access settings were changed. Research outputs remain separate from native Core and G1 archives.

## Infrastructure

The existing RTX 6000 Ada has 49,140 MiB total and approximately 14,064 MiB free at audit time. G1 and Core/InterGen authenticated health checks returned ready, with no queued Core work. Fresh native probes run serially as temporary processes; existing workers are not stopped or reloaded. A larger GPU does not fix incorrect skeleton mapping or skin weights.

## Result of this recovery

This is a reviewable research milestone, not an accepted studio feature. The exact original handshake and embrace were recovered, and fresh 210-frame inference reproduced their joint and feature arrays byte for byte. Direct native22 skinning uses the authored Xbot mesh, weights and inverse bind matrices, with source endpoints preserved. It bypasses Core27 and the approximate generated-character rig entirely. Bone-length variation still follows the source; this is not a rigid-body or contact solver.

Eight actual GPU samples were run: six InterGen scenes and two InterMask scenes. InterGen inference took approximately 0.87–0.95 seconds per sample (8–10 seconds including temporary process startup). InterMask inference took 0.75/0.56 seconds. Existing workers remained healthy. See the health report and InterMask run summary for commands, memory and source provenance.

The recovered handshake is the strongest semantic result: approach, sustained left-left wrist proximity, and separation. Its source wrists remain within 15 cm for 96 frames (3.2 seconds), which is proximity, not proof of clasped fingers. Full playback was reviewed. The optional authored finger curl is visibly subtle and does not solve palm contact; it is explicitly manual articulation, not model output. Native rotation twist priors did not establish a clear visual improvement and remain optional. Neither is promoted as a solved handshake.

The embrace compresses bodies late; the waltz has crossed/sliding feet and loose contact; the help-up trial does not clearly help a fallen partner stand. The dodge reads as loose sparring with poor range rather than a convincing timed attack. The high-five sample instead resembles separate boxing gestures and never completes a high-five. Dodge and high-five were screened using complete-sequence contact sheets. These remain rejected research samples. The InterMask handshake misses contact (minimum wrist distance about 18 cm); its embrace deeply intersects. Two samples do not establish a general model ranking, but neither deserves promotion.

The actual generated-city scene is reused as a backdrop. These samples do not navigate or collide with that city. Floor placement, foot sliding, surface contact and mesh penetration remain limitations. No clips are spliced and no actor roots are independently offset.

## Reproduce the viewer and capture

Run from this checkout. Use the project's existing Python environment. The asset is the official Three.js Xbot GLB at `https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Xbot.glb`; SHA256 `002f8d269de68e5dce3d25195caf390d1aa359bbfaae3fcf4c8dc78ec36c3ba5`. Save it to `.runtime/paired-rig-assets/Xbot.glb` and verify its hash. It is not vendored into this change.

```sh
.venv/bin/python experiments/native_pair_review.py \
  --input review/two-character/native-recovery/originals/handshake_seed42.npz \
  --rig-asset .runtime/paired-rig-assets/Xbot.glb \
  --mode 'Authored rig' --scene review/scene-integration/live-city.json \
  --port 2373 --capture-dir /tmp/stagezero-native-capture
```

Open localhost:2373 and press **Capture all 30 fps frames**. The capture writes all 210 frames as a seven-second MP4 plus source hashes, provenance, first/last frames and a contact sheet. Use a new capture directory for each run. Add `--handshake-fingers --contact-joints 20 20` only to reproduce the explicitly authored finger trial. Add `--native-rotation-prior` separately to reproduce that optional audit experiment.

```sh
.venv/bin/python -m unittest discover
.venv/bin/python experiments/measure_native_pair.py --help
.venv/bin/python experiments/intermask_probe.py --help
```

845 Python tests passed. These establish mechanical correctness, not animation quality. GPU reproduction details and checkpoint provenance are in `review/two-character/native-recovery/health-report.json` and `intermask/README.md`.

The GPU has approximately 14 GiB free after experiments; a larger GPU is not required for these models. The Pod disk now has only about 719 MiB free. Expand its storage before adding more checkpoints. No extra Pod was rented, and no existing worker was restarted.
