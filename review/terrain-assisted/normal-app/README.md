# Normal-app terrain integration

This continuation integrates terrain-aware movement into `director_viewer.py` and the existing Core controls. These are fresh GPU generations submitted through the normal Studio UI, not the earlier separate-viewer replay. [Usage and supported scope](../../../docs/TERRAIN-AWARE.md).

## Temple: complete fresh UI run

The normal Scene recipe generated **Traversable temple**. Three separate Motion-panel submissions produced:

| Submission | Saved result | Frames |
| --- | --- | --- |
| `walk up temple stairs then cross temple bridge` | [Stairs and bridge](temple-stairs-bridge.core.stagezero.npz) | 200 |
| `open temple gate` | [Opened gate](temple-open.core.stagezero.npz) | 280 |
| Reopen the saved project, then `enter` | [Complete route](temple-complete.core.stagezero.npz) | 360 |

The take was played and sought in normal Studio. [Normal-app screenshot](temple-normal-ui.png), [complete 20 fps video](temple-performance.mp4), [contact sheet](temple-contact-sheet.png), and [capture provenance](temple-capture.json) are included. The close-up video renders the exact saved normal-app presentation without another retarget or IK pass; it is a recording of those saved transforms, not an independent generation.

[Independent archive audit](temple-audit.json) verifies that both native positions/rotations/features and display positions/rotations/stance remain array-exact across the 200→280→360 append boundaries. The final rig has under 1 micrometre FK error, maximum knee flexion 99.99°, and maximum adjacent local leg rotation 31.02°. Stored whole-action checks report zero sole/swept-envelope penetration and unsupported samples, with flat-stance slip below 0.000008 m/s. These are finite geometric checks, not a dynamics certificate.

## Different background and repeated commands

The **Industrial switchback** starter scene uses approach stairs, a turn and crate detour, an X-axis bridge, and a rotated automatic workshop door. Fresh normal-UI commands produced [stairs and bridge, 280 frames](industrial-stairs-bridge.core.stagezero.npz), then `open workshop door, enter` produced [480 frames](industrial-complete.core.stagezero.npz). [Archive audit](industrial-audit.json), [24-second video](industrial-performance.mp4), [contact sheet](industrial-contact-sheet.png), and [capture provenance](industrial-capture.json) record this separate route. Both final videos were inspected at normal speed.

After loading the final 360-frame temple take, two separate `walk 0.5 metres forward` submissions produced [400 frames](temple-forward-once.core.stagezero.npz) and [440 frames](temple-forward-twice.core.stagezero.npz). They used the committed displayed facing even with the playback cursor at frame zero. The [repeated-command audit](repeated-commands-audit.json) confirms exact native and display prefixes.

**Descent and difficult pivots remain experimental.** A saved industrial descent trial passes CPU repair, but the final fresh UI descent rejected `Stationary pivot reverses direction within one hold` and kept the prior 480-frame take. Descent is not part of the verified demo. Do not present this as universal arbitrary-background traversal.

## Ordinary motion and rejection behavior

A fresh ordinary 240-frame Core take was generated from the normal UI before terrain mode. Its positions, rotations, and all native feature channels remained exactly identical after terrain generation, save/load, a rejected jump, and toggling terrain off. [Baseline archive](ordinary-baseline.core.stagezero.npz) and [exact preservation check](ordinary-preservation.json).

The real UI rejected `jump onto the temple roof` with **Jumping across terrain is unsupported** and retained all 440 committed native and display frames. [Preservation check](unsupported-command-preservation.json). CPU tests additionally cover missing support, blocked routes, ambiguous targets, cancellation, stale results, project corruption, transformed scenes, and repeated continuation.

## Development failures retained in the record

- A descriptive stair alias initially selected a slab named “Level stair entry.” The resolver now filters stair aliases through detected rendered stair flights before ranking proximity. Exact references to a non-stair still reject.
- A task-owned worker initially loaded an older backend missing `coordinate_frames_y`. Only that worker was updated to the PR backend. Ordinary generation had already worked; no shared GPU process was changed.
- The fresh temple stairs-to-bridge boundary initially rejected four reach frames. The saved native result was used for CPU diagnosis. A bounded 2.09 cm adjustment to one new contact target solved the mismatch without moving the committed prefix or widening knee/rotation/contact thresholds. A subsequent fresh UI generation passed.
- The first industrial take exposed early toe-off across a riser, swing paths cutting an obstacle corner, and preserved native arm motion too close to the crate. The solver now checks full shoe envelopes when choosing new contacts and follows route tangents for turning swings. Route planning reserves wider upper-body clearance while retaining the lower-leg allowance needed to approach stair risers. The failed native checkpoint was used for diagnosis rather than repeated GPU inference; its arm collision remains a valid rejection.

The archive's historical `accepted=false` and `runtime_publishable=false` flags distinguish this explicit assisted display from approval as ordinary native ARDY motion. Normal Studio publishes it only through the opt-in terrain-aware mode; ordinary generation never consumes the assisted poses.

## Reproduce

Use a matching PR checkout on the UI host and Core backend. The backend must support the optional terrain-height frame field. Start the normal app against an existing authorized Core service, then follow [the UI instructions](../../../docs/TERRAIN-AWARE.md). Saved playback needs no GPU:

```sh
PYTHONPATH=.:vendor/ardy python director_viewer.py --reference-only \
  --core-project review/terrain-assisted/normal-app/temple-complete.core.stagezero.npz
```

Capture the exact saved display separately with:

```sh
PYTHONPATH=.:vendor/ardy python experiments/capture_terrain_assisted.py \
  --core-project review/terrain-assisted/normal-app/temple-complete.core.stagezero.npz \
  --output /tmp/terrain-review --follow-camera
```

Open the printed capture page and click **Capture actual-rig proof**. The output contains every saved frame at 20 fps.

## Final validation

[Targeted test log](targeted-tests.log): 174 tests passed across session isolation, terrain motion, geometry, UI controls, archives, runtime/client/backend, scene recipes, and AI generation guidance. Another 15 UI-control tests and viewer compilation passed after the final wording/terrain-floor visibility adjustment. The AI guidance uses fake-gateway tests; live AI background generation was not verified.

The final display explicitly assists global facing as well as legs/pelvis, retaining upper-body relative articulation and exact native history. Archive `visual_review=pending` is capture-time metadata; this evidence index records subsequent human-facing visual inspection without rewriting the saved archives. Codex workers handled bounded implementation and verification; Neon GPT-6 Astra reviewed session invariants and the heading approach.
