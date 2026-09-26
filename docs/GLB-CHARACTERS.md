# GLB characters

The Studio can import a character without changing the ARDY checkpoint or motion
history. GLB geometry, materials and skinning stay in the browser; G1 joint poses
are retargeted locally. The original G1 character remains available.

## Use GLBs with prompts and live generation

Run the normal Studio to control GLBs with text prompts. With the existing
private recording, token and backend already configured at their default paths:

```sh
.venv/bin/python director_viewer.py --port 2341 --glb /path/to/character.glb --environment studio
```

Use a free port or stop only your own existing viewer before starting this.
The **Character** tab imports/selects models; **Direct** contains the motion
prompt, duration, **Generate**, extend and replace controls. **Takes** and the
timeline review generated motion. Generation automatically switches the project
to **Live ARDY** and applies the resulting G1 motion to the selected rigged GLB.
It returns complete generated segments, not token-by-token animation streaming.
The synthetic lab's standing/sweep pose is not substituted for generated motion.

`--recording PATH`, `--token-path PATH` and `--backend-url URL` can point to an
existing private setup or a separately owned SSH tunnel without modifying other
viewers. Defaults remain `assets/recorded_g1.csv`, `.runtime/api-token` and
`http://127.0.0.1:8765`. Credentials remain server-side. `--characters DIRECTORY`
selects the imported-model library; its default is `.runtime/characters`.
Studio reflection lighting defaults to `studio`; `warehouse` and `none` are
also supported. Starting the viewer does not provision or restart the backend.

## Run the isolated character lab

From this checkout, using its own environment:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-live.txt
git submodule update --init --recursive
cd studio_client
npm ci
npm run build
cd ..
.venv/bin/python examples/create_glb_demo_models.py
.venv/bin/python asset_viewer.py --port 2341 --glb .runtime/glb/demo-mixamo.glb
```

Open `http://127.0.0.1:2341`. The lab requires no token, private CSV or Pod. Its
joint sweep is deterministic **synthetic test motion**, not ARDY inference.
**Stand pose** uses a separate FK standing input with relaxed forearms, ankles
under the hips and approximately 12 degrees of forward knee flexion. Hip roll
compensates for the robot's lateral motor offsets; ankle counter-rotation keeps
the feet level. This lab pose was checked against the actual TASM mesh silhouette,
not only joint-center alignment. The sweep
opens both arms outward and returns along the same path; it does not deliberately
move them across the torso. The robot's identity-local calibration pose remains
the production retargeting reference. The lab places its floor at the standing
mesh's lowest rendered vertex once per character selection, keeping it fixed
during playback. This presentation alignment is not foot-contact IK.
It generates two simple textured test figures with different proportions and
G1/Mixamo names. These are technical fixtures, not production character artwork.
Use an unused port and separate runtime folder when sharing a machine.
The lab uses studio environment lighting so metallic and other PBR materials
have reflections. Use `--environment warehouse` for another preset or
`--environment none` for the earlier direct-light-only view.
After rebuilding the client, restart your own viewer process so Viser serves the
new asset index. Keep other workers' viewer processes running.

The lab is for inspecting materials and retargeting. Use the normal Studio above
for prompts and live generation. No Pod upload endpoint or backend contract was added.

## Import and select

1. Click **Load GLB**. The browser accepts one nonempty `.glb` up to 32 MiB.
2. The server validates it and stores it under its content hash in the private
   character library. Source filenames are not used as filesystem paths.
3. **Ready for motion** means a recognized and structurally validated humanoid
   mapping is available. **Mapping required** needs a JSON mapping. **Static
   preview** has no skin and can be inspected but cannot play/generate motion.
4. For a custom rig, choose the model and click **Load rig mapping** (JSON,
   maximum 1 MiB). See [rig mapping reference](../rig_profiles/README.md).
5. Use **Frame character** to fit the viewport, and the **Character** dropdown
   to return to G1 or another imported model.

The previous character remains visible until the initiating browser confirms the
new model has parsed and its shaders have compiled. Failed/stale loads do not
replace it. Each other tab switches after its own load; tabs share character
selection and motion state. Static activation pauses playback and invalidates
pending generation, preserving recorded takes. Switching between compatible
animated characters preserves the source motion and playback state.

Imported files and mappings persist locally; the current selection starts at G1
on a new server run unless `director_viewer.py --glb` or `asset_viewer.py --glb`
selects a startup model.
Reimporting identical bytes reuses the saved character, including its custom
mapping and name, and does not consume another library slot.
Character artwork is not embedded in exported motion project files. To use an
imported model on another viewer machine, import it there as well.

## Supported subset

- Core, uncompressed GLB 2.0 with an embedded buffer and embedded PNG/JPEG
  textures, triangle mesh primitives and standard glTF materials.
- At least one visible mesh reachable from the selected glTF scene.
- For motion: one connected humanoid skeleton, valid inverse binds and up to
  four nonnegative normalized influences per vertex; multiple meshes/skins can
  share this hierarchy. Positive uniform node scales and proper rotations.
- Conservative G1 and Mixamo name detection, or an explicit role-to-node mapping.
  Names alone do not certify animation quality; inspect a neutral pose, arms,
  knees and a walking clip before relying on an unfamiliar model.
- Models should already be exported in the intended glTF Y-up/metre convention.
  Root displacement is scaled by relative leg length. Model size is not silently
  changed to match the robot.

This first implementation rejects extension payloads/declarations (including
Draco, Meshopt, KTX2, GPU instancing and material extensions), sparse accessors,
morph targets, external resources, non-triangle primitives and unsupported rigs.
Rejecting optional extensions is intentional: silently processing them could
bypass validated geometry/resource limits. Export a core GLB variant for now.
Embedded animation clips do not auto-play in the controlled character renderer.

The importer caps file/JSON size, decoded accessors/images, hierarchy counts and
displayed geometry. Those are resource ceilings, not a claim that a 4-million-
triangle character sustains 25 FPS. Prefer roughly 50–100k triangles for the first
real model and measure on the target browser. The local library holds up to 16
models; choose a fresh directory for a separate test collection.

Retargeting aligns limb directions as well as rotation changes, so a human T/A
bind is not mistaken for the robot's neutral stance. Target bone lengths remain
intact. There is no foot-lock IK, physical contact correction, twist distribution,
face/finger animation or general quadruped support. The diagnostics disclose
ambiguous missing limb endpoints; mappings can supply explicit local bone axes.

## Verification

```sh
.venv/bin/python -m unittest discover -v -p 'test_*.py'
cd studio_client
npm run build
npm test
```

Tests cover malformed/resource-heavy GLBs, selected-scene and skinned bounds,
transport ownership/limits/retry, stale browser acknowledgements, static-mode
inference suspension, two different rig profiles, limb directions, root travel,
target lengths, frustum culling and interrupted browser loads.

The shared live app, inference service and other workers' environments must be
coordinated separately before deploying. Local tests and synthetic motion are not
evidence of real ARDY quality or production performance. Detailed implementation
decisions are in [the integration plan](GLB-INTEGRATION-PLAN.md).

### Local acceptance run — 2026-09-26

- Studio client: 11 Node tests passed; TypeScript check and production build
  completed. Browser retry/culling regressions are included in the test command.
- Full Python suite: 192 tests passed, including existing controller/backend
  regressions. The tests use local fixtures and do not call shared inference.
- Browser: both textured G1-named and Mixamo-named demo GLBs reached **Ready for
  motion**; frame 37 showed shoulder movement and root travel. Swapping models
  preserved the 1.48-second playhead, and selecting G1 restored the original mesh.
- A real WebSocket transport test rejected a 32 MiB + 1 byte declaration, then
  accepted the 23,784-byte demo GLB on the same connection. Stored bytes matched
  its hash. Browser selection separately exercised the real load/shader-ready
  acknowledgement; the transport test did not fabricate that acknowledgement.
- Browser file selection uploaded the demo GLB through **Load GLB** and reached
  **Ready for motion**, including the browser's real shader-ready acknowledgement.
- A skinless GLB reached **Static preview** and disabled the lab's play button
  without stopping the viewer. Viser button groups do not support disabling;
  the main Studio hides its grouped playback controls in this mode instead.
- Small-fixture retargeting measurements were approximately 2.3 ms median and
  4.2 ms p95 on this machine. These 16-node/180-triangle fixtures do not establish
  performance or deformation quality for production artwork.

### TASM pose regression

Real-character review exposed two gaps in the initial tests: terminal hip/shoulder
motor positions were used as anatomical centers, and the original signed sweep
deliberately crossed inward. Position landmarks now use hip/shoulder pitch while
rotation still comes from the complete motor chain. Independent G1 anatomy tests
cover this distinction and terminal-axis twist. A second visual review exposed
that sagittally collinear bones still produced wide legs and a backward-looking
knee silhouette on TASM. Lab tests now check hip-relative ankle width, positive
knee flexion, level feet, relaxed forearms, full-clip outward clearance and FK
consistency. Actual mesh front/side inspection remains necessary: joint tests
alone do not certify the appearance of an arbitrary character's skinning.
The TASM asset itself does not require another export for these motion fixes.
