> **Visually rejected experiment.** The user rejected this presentation: rigid body motion, unconvincing assisted flight/pickup, and poor character appearance. Its numerical checks do not establish animation quality. Preserve it as diagnostic history; do not treat it as a production-ready or accepted demo. Current work investigates full native body motion on generated human characters.

# Live web-swing experiment

This isolated prototype uses the existing generated city output, ARDY Core body
articulation, explicit web/flight controls, and authored paired IK. It does not
replace the studio, character generator, object pipeline, or background tools.

## Run

Initialize the existing ARDY submodule and use the installed StageZero Python
environment (`requirements-preview.txt`). The browser imports Three.js from the
existing studio dependencies; run `npm ci` in `studio_client` if needed.

```sh
git submodule update --init vendor/ardy
export STAGEZERO_PYTHON=/absolute/path/to/.venv/bin/python
export STAGEZERO_TOKEN_FILE=/absolute/path/to/private/api-token
export STAGEZERO_THREE_DIR=/absolute/path/to/studio_client/node_modules/three/build
export STAGEZERO_REALTIME_URL=http://127.0.0.1:8769
./run-swing.command
```

The existing authenticated realtime worker must be reachable through its private
SSH tunnel. No resource is provisioned by the launcher. Open
`http://127.0.0.1:2360/`. Click **Reset**, **Record**, **Start swing**, then steer
left and right while airborne. Request **Carry MJ**, wait for the physical
pickup, steer again, request **Land**, wait for grounded dismount, and click
**Kiss**. **Camera** cycles wide, chase and close views. **Stop & save** uploads
the uncut browser canvas recording, including the controls/HUD, to the run folder.

Reset before recording a second measured take. After a capture is saved, its report
and frame log are sealed; playback stays available without growing the log indefinitely.

Each run is stored under `.runtime/swing/<run-id>/` with continuous frame poses,
command timestamps, live native Core outputs, a report, and the actual browser
WebM. The browser also downloads the recording. Command response is measured
from server receipt to the browser's first paint acknowledgment; native inference
latency is measured separately. A control affects the next physics tick while
Core inference proceeds asynchronously. Playback continues with the last valid
body source while a new native window is generated.

## Source and methods

The city is a checksum-pinned copy of `review/scene-refinement/strong-city.json`.
It contains model-generated building assets composed by the existing background
pipeline. The adapter imports the actual mesh triangles and derives roof planes,
conservative collision bounds and facade anchors. Nothing is replaced with a
new hand-built city. See [background provenance](SWING-BACKGROUND.md).

Core's text-only swing probe did not produce flight. The implemented travel is
therefore a hybrid of gravity, unilateral tether constraints, bounded steering,
and authored lift/path assistance. Pickup and landing use obstacle-clear routes;
this is not a learned Spider-Man controller. Native Core contributes bounded
body articulation; authored limb IK owns the web hand, carried pose and grips.
MJ's root and heading move continuously through pickup and dismount. These
constraints hold a chosen piggyback pose; they do not predict balance or soft
body contact. The ending uses authored head targeting, mask reveal, eye and lip
animation. It is not generated facial acting.

The characters use the existing CoreSkin rig with experimental procedural
costumes. The ongoing generated-character system is untouched; its current G1
weights are not silently treated as a Core27-compatible facial rig.

## Model and resource boundaries

See [actual model probes and licenses](SWING-MODEL-RESEARCH.md). The live run uses
ARDY Core only; it does not depend on InterGen's noncommercial paired model.
The original G1 and realtime workers remain running. No additional Pod was
rented. This code is an experimental review prototype, not a general physical
interaction or commercial character-asset release.

## Production integration work

Keep the existing realtime scheduler's immutable committed prefix. Add a
versioned scene-affordance layer and a constrained flight/carry action provider
behind an explicit capability flag; publish its final canonical poses through
the same renderer and save/load format. Persist controller, attachment and
random state with the timeline, then test pause/seek/reconnect and action
interruptions. Background edits need revalidation of anchors and routes before
commit. A generated asset alone does not establish structural anchor strength.

High-quality human carrying and kissing require authored motion capture or a
contact-aware paired dataset, retargeting to the actual character rigs, finger
IK, facial blendshapes, and a supported commercial model/license. Collision
proxies need mesh-level continuous collision and body-contact handling before
shipping unrestricted interaction. The fixed miniature city and curated action
sequence are evaluation limits, not universal navigation claims.
